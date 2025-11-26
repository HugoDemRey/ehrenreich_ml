from typing import List, Optional, Tuple
import csv

class TSAnnotations:

    @staticmethod
    def load_annotations(csv_path: str) -> List[Tuple[float, float, str]]:
        """
        Read time annotations from a semicolon-separated CSV-like file.
        Handles optional header and quoted fields, e.g.
        "start";"end";"label"
        0;87;"A"
        87.1;338;"B"
        768.1;900;""
        """

        annotations: List[Tuple[float, float, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=";", quotechar='"')
            for row in reader:
                if not row:
                    continue
                # Expect at least two columns for start and end
                if len(row) < 2:
                    continue
                start_s, end_s = row[0].strip(), row[1].strip()

                # Skip header or malformed rows where start/end are not numbers
                try:
                    start_time = float(start_s)
                    end_time = float(end_s)
                except ValueError:
                    continue

                label = ""
                if len(row) > 2:
                    label = row[2].strip()
                    # csv.reader will already remove surrounding quotes, but normalize empty quoted fields
                    if label == '""':
                        label = ""
                annotations.append((start_time, end_time, label))
        return annotations
    

    @staticmethod
    def sub_annotations(annotations: List[Tuple[float, float, str]], start: float, end: float) -> List[Tuple[float, float, str]]:
        """
        Return a sublist of annotations that fall within the specified time range [start, end].
        """
        filtered = [ann for ann in annotations if ann[0] >= start and ann[1] <= end]
        offset_filtered = [(ann[0] - start, ann[1] - start, ann[2]) for ann in filtered]
        return offset_filtered
    
    @staticmethod
    def load_transitions_txt(txt_path: str) -> List[float]:
        """
        Load transition timestamps from a text file, one timestamp per line.
        """
        transitions: List[float] = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    time_stamp = float(line)
                    transitions.append(time_stamp)
                except ValueError:
                    continue
        return transitions
    
    @staticmethod
    def save_transitions(json_path: str, transitions: List[float]):
        """
        Save transition timestamps to a text file, one timestamp per line.
        """
        import json
        with open(json_path, "w", encoding="utf-8") as jp:
            json.dump(transitions, jp)

    @staticmethod
    def load_transitions(json_path: str) -> Optional[List[float]]:
        """
        Load transition timestamps from a JSON file.
        """
        import json
        try:
            with open(json_path, "r", encoding="utf-8") as jp:
                transitions = json.load(jp)
            return transitions
        except (FileNotFoundError, json.JSONDecodeError):
            return None