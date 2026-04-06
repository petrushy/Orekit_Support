# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "orekit-jpype>=13.1.4.0",
#     "orekitdata @ git+https://gitlab.orekit.org/orekit/orekit-data.git",
# ]
# ///

"""Read a CCSDS OMM or OEM XML file, propagate it with Orekit, and write a fitted TLE.

OMM workflow:
1. Parse the OMM XML with Orekit.
2. Let Orekit create the corresponding initial TLE and SGP4 propagator.
3. Build a bounded ephemeris from the OMM epoch to +3 days with 20-minute sampling.
4. Fit one new TLE to those sampled states, keeping the TLE epoch equal to the OMM epoch.
5. Write the fitted TLE to a sibling `.TLE` file.

OEM workflow:
1. Parse the OEM XML with Orekit — it already contains a tabulated ephemeris.
2. Use the OEM's built-in bounded propagator directly (no SGP4 step needed).
3. Fit one new TLE to states sampled from the OEM's full time range.
4. Write the fitted TLE to a sibling `.TLE` file.

Orekit APIs used here:
- `ParserBuilder().buildNdmParser().parseMessage(...)` for CCSDS OMM/OEM parsing
- `Omm.generateTLE()` for the initial TLE (OMM path only)
- `TLEPropagator.selectExtrapolator(...)` for the SGP4 propagator (OMM path only)
- `getEphemerisGenerator()` for the bounded ephemeris (OMM path only)
- `EphemerisFile.SatelliteEphemeris.getPropagator()` for the OEM ephemeris propagator
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
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import Constants

STEP_SECONDS = 20 * 60
DURATION_SECONDS = 3 * 24 * 60 * 60
FIT_POSITION_ONLY = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CelesTrak OMM or OEM XML file and fit a TLE to it."
    )
    parser.add_argument(
        "orbitdescription",
        type=Path,
        help="Path to the OMM or OEM XML file.",
    )
    parser.add_argument(
        "--type",
        choices=["omm", "oem"],
        required=True,
        help="Type of the input file: 'omm' for an Orbit Mean-elements Message, "
             "'oem' for an Orbit Ephemeris Message.",
    )
    parser.add_argument(
        "--norad",
        type=int,
        default=None,
        metavar="ID",
        help="Override the NORAD catalog number written into the output TLE "
             "(e.g. --norad 25544).  Required when using --type oem, which "
             "carries no catalog number.",
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


def _make_data_source(xml_content: str, name: str) -> DataSource:
    """Wrap an XML string in the DataSource Orekit expects."""
    reader_opener = jpype.JProxy(
        DataSource.ReaderOpener,
        dict(openOnce=lambda: StringReader(xml_content)),
    )
    return DataSource(name, reader_opener)


def create_propagator_from_omm_xml(omm_xml: str):
    """Parse one OMM message and return ``(omm, original_tle, propagator)``."""
    omm_xml = normalize_omm_xml(omm_xml)

    # `DataSource.ReaderOpener` is Orekit's text-input hook. JPype's `JProxy`
    # lets us implement that small Java interface directly from Python.
    ndm_message = ParserBuilder().buildNdmParser().parseMessage(_make_data_source(omm_xml, "omm.xml"))
    omm_messages = list(ndm_message.getConstituents())
    if len(omm_messages) != 1:
        raise ValueError(f"Expected exactly one OMM constituent, got {len(omm_messages)}")

    omm = omm_messages[0]
    tle = omm.generateTLE()
    propagator = TLEPropagator.selectExtrapolator(tle)
    return omm, tle, propagator


def normalize_oem_xml(oem_xml: str) -> str:
    """Remove empty COMMENT elements that cause Orekit's parser to crash.

    Some OEM providers (e.g. NASA JSC) emit blank ``<COMMENT></COMMENT>`` lines
    as visual separators.  Orekit calls ``getContentAsNormalizedString()`` on
    every COMMENT token and throws a NullPointerException when the content is
    empty.  Stripping them before parsing is the safest fix.
    """
    return re.sub(r"<COMMENT>\s*</COMMENT>\s*\n?", "", oem_xml)


def create_ephemeris_from_oem_xml(oem_xml: str):
    """Parse one OEM message and return ``(object_name, object_id, ephemeris)``.

    An OEM already contains a tabulated ephemeris, so the returned ``ephemeris``
    is a ``BoundedPropagator`` that can be sampled directly — no SGP4 step needed.

    ``withMu`` supplies Earth's gravitational parameter (µ), which the OEM format
    does not carry but Orekit needs to build the interpolating propagator.
    """
    oem_xml = normalize_oem_xml(oem_xml)
    ndm_message = (
        ParserBuilder()
        .withMu(Constants.WGS84_EARTH_MU)
        .buildNdmParser()
        .parseMessage(_make_data_source(oem_xml, "oem.xml"))
    )
    oem_messages = list(ndm_message.getConstituents())
    if len(oem_messages) != 1:
        raise ValueError(f"Expected exactly one OEM constituent, got {len(oem_messages)}")

    oem = oem_messages[0]
    satellites = oem.getSatellites()
    sat_keys = list(satellites.keySet())
    if len(sat_keys) != 1:
        raise ValueError(f"Expected exactly one satellite in OEM, got {len(sat_keys)}")

    segment_metadata = list(oem.getSegments())[0].getMetadata()
    object_name = segment_metadata.getObjectName()
    object_id = segment_metadata.getObjectID()

    ephemeris = satellites.get(sat_keys[0]).getPropagator()
    return object_name, object_id, ephemeris


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


def build_reference_tle_from_state(state, source_tle=None, norad_id=None):
    """Build Orekit's required TLE reference guess from one spacecraft state.

    The Keplerian orbital elements come from ``state``. Satellite identification
    metadata is copied from ``source_tle`` when provided; placeholder values are
    used otherwise (e.g. when fitting from an OEM that carries no NORAD number).
    If ``norad_id`` is given it overrides whatever ``source_tle`` carries.
    """
    orbit = OrbitType.KEPLERIAN.convertType(state.getOrbit())

    if source_tle is not None:
        satellite_number = source_tle.getSatelliteNumber()
        classification = source_tle.getClassification()
        launch_year = source_tle.getLaunchYear()
        launch_number = source_tle.getLaunchNumber()
        launch_piece = source_tle.getLaunchPiece()
        ephemeris_type = source_tle.getEphemerisType()
        element_number = source_tle.getElementNumber()
        revolution_number = source_tle.getRevolutionNumberAtEpoch()
        utc = source_tle.getUtc()
    else:
        # OEM files carry no NORAD catalog number; use placeholder values.
        satellite_number = 0
        classification = "U"
        launch_year = 0
        launch_number = 0
        launch_piece = "A"
        ephemeris_type = 0
        element_number = 0
        revolution_number = 0
        utc = TimeScalesFactory.getUTC()

    if norad_id is not None:
        satellite_number = norad_id

    return TLE(
        satellite_number,
        classification,
        launch_year,
        launch_number,
        launch_piece,
        ephemeris_type,
        element_number,
        state.getDate(),
        orbit.getKeplerianMeanMotion(),
        0.0,
        0.0,
        orbit.getE(),
        orbit.getI(),
        orbit.getPerigeeArgument(),
        orbit.getRightAscensionOfAscendingNode(),
        orbit.getMeanAnomaly(),
        revolution_number,
        0.0,
        utc,
    )


def fit_new_tle_from_ephemeris(ephemeris, start_date, end_date, step_seconds, source_tle=None, norad_id=None):
    """Fit one new TLE to sampled ephemeris states.

    This follows Orekit's TLE fitting route:
    - build a reference TLE near the desired epoch
    - feed sampled states to `FiniteDifferencePropagatorConverter`
    - ask Orekit to fit using TLE dynamics

    ``useOnlyPosition=True`` means the fitting is driven by position, which gives
    a meaningful RMS error in metres.  When ``source_tle`` is ``None`` (OEM path),
    placeholder satellite identification metadata is used in the template TLE.
    ``norad_id`` overrides the catalog number in the output TLE when provided.
    """
    states = collect_states(ephemeris, start_date, end_date, step_seconds)
    reference_state = ephemeris.propagate(start_date)
    template_tle = build_reference_tle_from_state(reference_state, source_tle, norad_id)
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
    xml_content = load_orbit_description(args.orbitdescription)
    file_type = args.type

    if file_type == "omm":
        omm, source_tle, propagator = create_propagator_from_omm_xml(xml_content)
        object_name = omm.getMetadata().getObjectName()
        object_id = omm.getMetadata().getObjectID()
        epoch = omm.getData().getKeplerianElementsBlock().getEpoch()
        end_date = epoch.shiftedBy(DURATION_SECONDS)
        ephemeris = build_bounded_ephemeris(propagator, epoch, end_date)
    else:  # oem
        object_name, object_id, ephemeris = create_ephemeris_from_oem_xml(xml_content)
        source_tle = None
        epoch = ephemeris.getMinDate()
        end_date = ephemeris.getMaxDate()

    regenerated_tle, regenerated_tle_rms, sample_count = fit_new_tle_from_ephemeris(
        ephemeris, epoch, end_date, STEP_SECONDS, source_tle, args.norad
    )
    output_path = tle_output_path(args.orbitdescription)
    write_tle_file(output_path, regenerated_tle)

    print(f"Loaded {file_type.upper()} from {args.orbitdescription}")
    print(f"Object: {object_name} ({object_id})")
    print(f"Epoch: {epoch}")
    print(f"Ephemeris start: {ephemeris.getMinDate()}")
    print(f"Ephemeris end: {ephemeris.getMaxDate()}")
    print(f"Step seconds: {STEP_SECONDS}")
    print(f"Samples: {sample_count}")
    if source_tle is not None:
        print(f"Original TLE line 1: {source_tle.getLine1()}")
        print(f"Original TLE line 2: {source_tle.getLine2()}")
    print(f"Regenerated TLE epoch: {regenerated_tle.getDate()}")
    print(f"Regenerated TLE RMS: {regenerated_tle_rms:.3f} m")
    print(f"Regenerated TLE line 1: {regenerated_tle.getLine1()}")
    print(f"Regenerated TLE line 2: {regenerated_tle.getLine2()}")
    print(f"TLE file: {output_path}")


if __name__ == "__main__":
    main()
