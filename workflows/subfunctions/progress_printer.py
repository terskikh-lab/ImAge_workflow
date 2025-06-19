import time
import os

def progress_printer(progress_dict, progress_name, total):
    """
    Prints the progress of multiple tasks.

    This version prints updates as new lines, allowing the user to scroll up
    and view previous states, unlike the original which cleared the screen.

    Args:
        progress_dict (dict): A dictionary where keys are task names (e.g., thread IDs)
                              and values are their status (e.g., "running", "done", "skipped").
        progress_name (str): A name for the overall progress being tracked.
        total (int): The total number of tasks to complete.
    """
    # Keep track of the last state that was printed to avoid redundant output
    # if the progress hasn't changed.
    last_printed_state = {}
    first_print = True

    while True:
        current_lines = []
        done_count = 0
        current_progress_status = {}

        # Collect current status for all tasks, sorted by key for consistent display
        for key in sorted(progress_dict.keys()):
            status = progress_dict[key]
            current_lines.append(f"{key}: {status}")
            current_progress_status[key] = status

            if status in ["done", "skipped"]:
                done_count += 1

        # Only print an update if the progress state has changed
        # or if it's the very first time printing.
        if current_progress_status != last_printed_state or first_print:
            # Add a separator and timestamp to clearly distinguish updates
            print("\n" + "=" * 50)
            print(f"Progress ({progress_name}) - Update at {time.strftime('%Y-%m-%d %H:%M:%S')}:")
            print("=" * 50)
            print("\n".join(current_lines))
            last_printed_state = current_progress_status.copy()
            first_print = False

        # Check if all tasks are done
        if done_count == total:
            print("\n" + "=" * 50)
            print(f"Progress ({progress_name}): All tasks completed!")
            print("=" * 50 + "\n")
            break # Exit the loop once all tasks are done

        time.sleep(1) # Wait for 1 second before checking for updates again
