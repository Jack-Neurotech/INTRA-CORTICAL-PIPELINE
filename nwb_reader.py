
# ==========================================
# INTRACORTICAL BCI DECODER
#
# NWB FILE INGESTION LAYER
#
# PURPOSE:
#
# 1. Open a file-selection dialog
# 2. Start in the user's Downloads folder
# 3. Allow the user to select an NWB file
# 4. Validate the selected file
# 5. Load the NWB recording
# 6. Return the NWBFile object
#
# INPUT:
#     User-selected .nwb file
#
# OUTPUT:
#     NWBFile object
# ==========================================

from pathlib import Path
import subprocess

from pynwb import NWBHDF5IO


class NWBReader:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(self):

        self.file_path = None

        self.io = None

        self.nwbfile = None


    # ==========================================
    # SELECT NWB FILE
    #
    # Opens the macOS file picker.
    #
    # Initial location:
    #     ~/Downloads
    # ==========================================

    def select_file(self):

        downloads_folder = (
            Path.home()
            / "Downloads"
        )


        applescript = f'''
        tell application "Finder"

            activate

            set selectedFile to choose file ¬
                with prompt "Select an intracortical NWB recording" ¬
                default location POSIX file "{downloads_folder}" ¬
                of type {{"nwb"}}

            return POSIX path of selectedFile

        end tell
        '''


        result = subprocess.run(

            [
                "osascript",
                "-e",
                applescript
            ],

            capture_output=True,

            text=True
        )


        # ======================================
        # USER CANCELLED
        # ======================================

        if result.returncode != 0:

            raise RuntimeError(
                "NWB file selection was cancelled."
            )


        selected_path = (
            result.stdout.strip()
        )


        if not selected_path:

            raise RuntimeError(
                "No NWB file was selected."
            )


        self.file_path = Path(
            selected_path
        )


        return self.file_path


    # ==========================================
    # VALIDATE FILE
    # ==========================================

    def validate(self):

        if self.file_path is None:

            raise ValueError(
                "No NWB file has been selected."
            )


        if not self.file_path.exists():

            raise FileNotFoundError(
                f"NWB file not found: {self.file_path}"
            )


        if self.file_path.suffix.lower() != ".nwb":

            raise ValueError(
                "Invalid file type. Expected an .nwb file."
            )


    # ==========================================
    # LOAD NWB FILE
    # ==========================================

    def load(self):

        # --------------------------------------
        # Select file if one has not already
        # been selected.
        # --------------------------------------

        if self.file_path is None:

            self.select_file()


        # --------------------------------------
        # Validate
        # --------------------------------------

        self.validate()


        # --------------------------------------
        # Open NWB file
        # --------------------------------------

        self.io = NWBHDF5IO(

            str(self.file_path),

            mode="r"

        )


        self.nwbfile = (
            self.io.read()
        )


        return self.nwbfile


    # ==========================================
    # CLOSE FILE
    # ==========================================

    def close(self):

        if self.io is not None:

            self.io.close()

            self.io = None
