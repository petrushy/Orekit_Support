# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "orekit-jpype>=13.1.4.0",
#     "orekitdata @ git+https://gitlab.orekit.org/orekit/orekit-data.git",
# ]
# ///

"""Read a CCSDS OMM XML file, propagate it with Orekit, and write a fitted TLE.

Workflow:
1. Parse the input OMM XML with Orekit.
2. Let Orekit create the corresponding initial TLE and SGP4 propagator.
3. Build a bounded ephemeris from the OMM epoch to +3 days with 20 minute sampling.
4. Fit one new TLE to those sampled states, while keeping the fitted TLE epoch equal
   to the original OMM epoch.
5. Write the fitted TLE to a sibling `.TLE` file.

Orekit APIs used here:
- `ParserBuilder().buildNdmParser().parseMessage(...)` for CCSDS OMM parsing
- `Omm.generateTLE()` for the initial TLE
- `TLEPropagator.selectExtrapolator(...)` for the SGP4 propagator
- `getEphemerisGenerator()` for the bounded ephemeris
- `TLEPropagatorBuilder` + `FiniteDifferencePropagatorConverter` for fitting a
  new TLE from sampled position states
  
This code was to a large degree written by AI (Codex + Claude) directed by Petrus Hyvönen 2026
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jpype
from orekit_jpype import initVM  # initVM needs to be run before any other java related imports
initVM()

from orekit_jpype.pyhelpers import setup_orekit_data

# The JVM must be started and Orekit data loaded before any Java classes are imported.
setup_orekit_data()

from java.io import StringReader
from java.util import ArrayList
from org.orekit.data import DataSource
from org.orekit.files.ccsds.ndm import ParserBuilder
from org.orekit.orbits import OrbitType, PositionAngleType
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.propagation.analytical.tle.generation import (
    LeastSquaresTleGenerationAlgorithm,
)
from org.orekit.propagation.conversion import (
    FiniteDifferencePropagatorConverter,
    TLEPropagatorBuilder,
)

STEP_SECONDS = 20 * 60
DURATION_SECONDS = 3 * 24 * 60 * 60
FIT_POSITION_ONLY = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CelesTrak OMM XML file and create an Orekit propagator."
    )
    parser.add_argument(
        "orbitdescription",
        type=Path,
        help="Path to the OMM XML file.",
    )
    return parser.parse_args()


def load_orbit_description(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Orbit-description file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Orbit-description path is not a file: {path}")

    return path.read_text(encoding="utf-8")


def normalize_omm_xml(omm_xml: str) -> str:
    """Fill a few header fields that Orekit expects to be present.

    CelesTrak OMM files sometimes export empty or self-closing tags for
    CREATION_DATE and ORIGINATOR.  Orekit's CCSDS parser rejects those, so
    we substitute sensible defaults before parsing.
    """
    epoch_match = re.search(r"<EPOCH>(.*?)</EPOCH>", omm_xml, re.DOTALL)
    creation_date = epoch_match.group(1).strip() if epoch_match else "2000-01-01T00:00:00"

    normalized = re.sub(
        r"<CREATION_DATE\s*/>",
        f"<CREATION_DATE>{creation_date}</CREATION_DATE>",
        omm_xml,
    )
    normalized = re.sub(
        r"<CREATION_DATE>\s*</CREATION_DATE>",
        f"<CREATION_DATE>{creation_date}</CREATION_DATE>",
        normalized,
    )
    normalized = re.sub(
        r"<ORIGINATOR\s*/>",
        "<ORIGINATOR>unknown</ORIGINATOR>",
        normalized,
    )
    normalized = re.sub(
        r"<ORIGINATOR>\s*</ORIGINATOR>",
        "<ORIGINATOR>unknown</ORIGINATOR>",
        normalized,
    )
    return normalized


def create_propagator_from_omm_xml(omm_xml: str):
    """Parse one OMM message and return `(omm, original_tle, propagator)`."""
    omm_xml = normalize_omm_xml(omm_xml)

    # `DataSource.ReaderOpener` is Orekit's text-input hook. JPype's `JProxy`
    # lets us implement that small Java interface directly from Python.
    reader_opener = jpype.JProxy(
        DataSource.ReaderOpener,
        dict(openOnce=lambda: StringReader(omm_xml)),
    )
    data_source = DataSource("omm.xml", reader_opener)
    ndm_message = ParserBuilder().buildNdmParser().parseMessage(data_source)
    omm_messages = list(ndm_message.getConstituents())
    if len(omm_messages) != 1:
        raise ValueError(f"Expected exactly one OMM constituent, got {len(omm_messages)}")

    omm = omm_messages[0]
    tle = omm.generateTLE()
    propagator = TLEPropagator.selectExtrapolator(tle)
    return omm, tle, propagator


def build_bounded_ephemeris(propagator, start_date, end_date):
    """Generate a bounded ephemeris by propagating once over the full interval."""
    generator = propagator.getEphemerisGenerator()
    propagator.propagate(start_date, end_date)
    return generator.getGeneratedEphemeris()


def collect_states(ephemeris, start_date, end_date, step_seconds: float):
    """Sample states from the bounded ephemeris, including the final endpoint."""
    states = ArrayList()
    current_date = start_date

    # `durationFrom` returns (current - end) in seconds; negative while current < end.
    while current_date.durationFrom(end_date) < 0:
        states.add(ephemeris.propagate(current_date))
        current_date = current_date.shiftedBy(step_seconds)

    states.add(ephemeris.propagate(end_date))
    return states


def build_reference_tle_from_state(state, source_tle):
    """Build Orekit's required TLE reference guess from one spacecraft state.

    The orbital elements come from the state itself, while identification metadata
    comes from the original TLE generated from the OMM.
    """
    orbit = OrbitType.KEPLERIAN.convertType(state.getOrbit())
    return TLE(
        source_tle.getSatelliteNumber(),
        source_tle.getClassification(),
        source_tle.getLaunchYear(),
        source_tle.getLaunchNumber(),
        source_tle.getLaunchPiece(),
        source_tle.getEphemerisType(),
        source_tle.getElementNumber(),
        state.getDate(),
        orbit.getKeplerianMeanMotion(),
        0.0,
        0.0,
        orbit.getE(),
        orbit.getI(),
        orbit.getPerigeeArgument(),
        orbit.getRightAscensionOfAscendingNode(),
        orbit.getMeanAnomaly(),
        source_tle.getRevolutionNumberAtEpoch(),
        0.0,
        source_tle.getUtc(),
    )


def fit_new_tle_from_ephemeris(ephemeris, start_date, end_date, step_seconds, source_tle):
    """Fit one new TLE to sampled ephemeris states.

    This follows Orekit's TLE fitting route:
    - build a reference TLE near the desired epoch
    - feed sampled states to `FiniteDifferencePropagatorConverter`
    - ask Orekit to fit using TLE dynamics

    `useOnlyPosition=True` means the fitting is driven by position, which gives more clarity on the
    RMS error (in meters).
    """
    states = collect_states(ephemeris, start_date, end_date, step_seconds)
    reference_state = ephemeris.propagate(start_date)
    template_tle = build_reference_tle_from_state(reference_state, source_tle)
    generation_algorithm = LeastSquaresTleGenerationAlgorithm()
    builder = TLEPropagatorBuilder(
        template_tle,
        PositionAngleType.MEAN,
        1.0,
        generation_algorithm,
    )
    fitter = FiniteDifferencePropagatorConverter(builder, 1.0e-4, 1000)
    fitted_propagator = fitter.convert(states, FIT_POSITION_ONLY, ["BSTAR"])
    fitted_tle = fitted_propagator.getTLE()
    return fitted_tle, fitter.getRMS(), states.size()


def tle_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".TLE")


def write_tle_file(path: Path, tle) -> None:
    path.write_text(f"{tle.getLine1()}\n{tle.getLine2()}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    omm_xml = load_orbit_description(args.orbitdescription)
    omm, tle, propagator = create_propagator_from_omm_xml(omm_xml)
    epoch = omm.getData().getKeplerianElementsBlock().getEpoch()
    end_date = epoch.shiftedBy(DURATION_SECONDS)
    ephemeris = build_bounded_ephemeris(propagator, epoch, end_date)
    regenerated_tle, regenerated_tle_rms, sample_count = fit_new_tle_from_ephemeris(
        ephemeris, epoch, end_date, STEP_SECONDS, tle
    )
    output_path = tle_output_path(args.orbitdescription)
    write_tle_file(output_path, regenerated_tle)

    print(f"Loaded OMM from {args.orbitdescription}")
    print(f"Object: {omm.getMetadata().getObjectName()} ({omm.getMetadata().getObjectID()})")
    print(f"Epoch: {epoch}")
    print(f"Ephemeris start: {ephemeris.getMinDate()}")
    print(f"Ephemeris end: {ephemeris.getMaxDate()}")
    print(f"Step seconds: {STEP_SECONDS}")
    print(f"Samples: {sample_count}")
    print(f"Original TLE line 1: {tle.getLine1()}")
    print(f"Original TLE line 2: {tle.getLine2()}")
    print(f"Regenerated TLE epoch: {regenerated_tle.getDate()}")
    print(f"Regenerated TLE RMS: {regenerated_tle_rms:.3f} m")
    print(f"Regenerated TLE line 1: {regenerated_tle.getLine1()}")
    print(f"Regenerated TLE line 2: {regenerated_tle.getLine2()}")
    print(f"TLE file: {output_path}")
    print(f"Propagator: {propagator.getClass().getName()}")


if __name__ == "__main__":
    main()
