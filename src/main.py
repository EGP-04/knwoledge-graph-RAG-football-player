# src/main.py
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.executor import execute

if __name__ == "__main__":
    while True:
        try:
            query = input("\nAsk: ")
            response = execute(query)
            print("\nAnswer:", response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[System Error] An unexpected error occurred: {e}")