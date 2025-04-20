import os
import json
import traceback


def log_error(message, exception=None):
    """Log detailed error information"""
    print(f"ERROR: {message}")
    if exception:
        print(f"Exception type: {type(exception).__name__}")
        print(f"Exception message: {str(exception)}")
        print("Traceback:")
        traceback.print_exc()


def inspect_directory(path, max_depth=2, current_depth=0):
    """Recursively inspect directory structure and contents"""
    if current_depth > max_depth:
        return

    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return

    if os.path.isfile(path):
        print(f"File: {path}")
        if path.endswith(".json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    print(f"  JSON keys: {list(data.keys())}")
                    if "frames" in data:
                        print(f"  Number of frames: {len(data['frames'])}")
                        if data["frames"]:
                            print(
                                f"  First frame keys: {list(data['frames'][0].keys())}"
                            )
            except Exception as e:
                print(f"  Error inspecting JSON: {e}")
        return

    print(f"Directory: {path}")
    try:
        contents = os.listdir(path)
        print(f"  Contains {len(contents)} items")
        for item in contents:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"  Subdirectory: {item}")
                inspect_directory(item_path, max_depth, current_depth + 1)
            else:
                print(f"  File: {item}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
