import time
def progress_printer(progress_dict, progress_name, total):
    while True:
        lines = []
        done = 0
        for key in sorted(progress_dict.keys()):
            status = progress_dict[key]
            lines.append(f"{key}: {status}")
            if status == "done" or status == "skipped":
                done += 1
        print("\033c", end="")  # Clear terminal
        print(f"Progress ({progress_name}):")
        print("\n".join(lines))
        if done == total:
            break
        time.sleep(1)