import time
import sys

def progress_printer(progress_dict, name, total):
    # Store the last status of each key to avoid redundant updates
    last_status = {}
    while True:
        done = 0
        processing = 0
        skipped = 0
        pending = 0
        changed_lines = []
        
        for key, status in progress_dict.items():
            if status.startswith("skipped"):
                skipped += 1
            elif status.startswith("processing"):
                processing += 1
            elif status.startswith("pending"):
                pending += 1
            elif status.startswith("segmenting") or status.startswith("loading"):
                processing += 1
            elif status.startswith("segmenting images... done"):
                done += 1
            else:
                done += 1
            
            # Check if the status has changed
            if last_status.get(key) != status:
                changed_lines.append(f"{key}: {status}")
                last_status[key] = status
        
        # Print only changed lines
        for line in changed_lines:
            sys.stdout.write(line + "\n")
        
        # Print summary line
        summary = (f"{name}: {done} done, {processing} processing, "
                   f"{skipped} skipped, {pending} pending / {total}")
        sys.stdout.write("\r" + summary)
        sys.stdout.flush()
        
        if done + skipped >= total:
            sys.stdout.write("\n")  # Move to the next line after finishing
            break
        
        time.sleep(0.5)
