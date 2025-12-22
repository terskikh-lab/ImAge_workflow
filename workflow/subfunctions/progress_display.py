import os
import time

class ProgressDisplay:
    """
    A class to display progress for multiprocessing tasks in the terminal.
    """
    def __init__(self, jobs, num_processes):
        self.jobs = jobs
        self.num_processes = num_processes
        self.total_jobs = len(jobs)
        self.statuses = {i: "pending" for i in range(self.total_jobs)}
        self.results = []
        self.start_time = None
        self.next_job_to_run = 0

    def start(self):
        """
        Initializes the display and starts the first batch of jobs.
        """
        self.start_time = time.time()
        for i in range(min(self.num_processes, self.total_jobs)):
            self.statuses[i] = "running"
            self.next_job_to_run += 1
        
        if self.total_jobs > 0:
            # Create space for the display
            for _ in range(self.num_processes + 2):
                print()

    def update(self, result):
        """
        Updates the status of a job and redraws the display.
        """
        if result:
            index = result['index']
            if 'error' in result:
                self.statuses[index] = "failed"
            else:
                self.statuses[index] = "done"
                self.results.append(result)

            if self.next_job_to_run < self.total_jobs:
                self.statuses[self.next_job_to_run] = "running"
                self.next_job_to_run += 1
        
        self._redraw()

    def finish(self):
        """
        Cleans up the display by moving the cursor to the end.
        """
        if self.total_jobs > 0:
            # Move cursor down to the end
            print(f"\033[{self.num_processes + 2}B", end="", flush=True)

    def _redraw(self):
        """
        Redraws the entire progress display.
        """
        try:
            terminal_height, terminal_width = os.get_terminal_size()
        except OSError:
            terminal_height, terminal_width = 24, 80

        # Move cursor to top of display area
        if self.total_jobs > 0:
            print(f"\033[{self.num_processes + 2}A", end="", flush=True)

        # Redraw active jobs section
        active_jobs = [self.jobs[i] for i, s in self.statuses.items() if s == "running"]
        header = f"--- Active Jobs ({len(active_jobs)}/{self.num_processes}) ---"
        print(f"\033[2K{header}".ljust(terminal_width))

        #-3 for header, summary, and potential message line
        max_visible_jobs = min(self.num_processes, terminal_height - 3) 

        for i in range(max_visible_jobs):
            if i < len(active_jobs):
                line = f"  {active_jobs[i]}"
            else:
                line = ""
            print(f"\033[2K{line}".ljust(terminal_width), flush=True)

        if len(active_jobs) > max_visible_jobs:
            remaining_jobs = len(active_jobs) - max_visible_jobs
            print(f"\033[2K--- and {remaining_jobs} more active jobs ---".ljust(terminal_width), flush=True)
        
        # Fill remaining allocated space with blank lines
        # The total space for jobs is self.num_processes.
        # We've used max_visible_jobs lines, and maybe 1 for the "more jobs" message.
        lines_used = max_visible_jobs
        if len(active_jobs) > max_visible_jobs:
            lines_used += 1
        
        for _ in range(self.num_processes - lines_used):
            print("\033[2K".ljust(terminal_width), flush=True)


        # Redraw summary
        jobs_done = len(self.results)
        
        if self.total_jobs > 0:
            progress_fraction = jobs_done / self.total_jobs
            percentage = int(progress_fraction * 100)

            bar_width = 40
            filled_length = int(bar_width * progress_fraction)
            bar = '█' * filled_length + '░' * (bar_width - filled_length)
            progress_bar_str = f"\033[92m{bar}\033[0m"
        else:
            percentage = 0
            progress_bar_str = ""

        if jobs_done > 0:
            elapsed_time = time.time() - self.start_time
            avg_time_per_job = elapsed_time / jobs_done
            jobs_remaining = self.total_jobs - jobs_done
            eta_seconds = int(jobs_remaining * avg_time_per_job)
            eta_str = f"{eta_seconds // 60:02d}m{eta_seconds % 60:02d}s"
        else:
            eta_str = "calculating..."
        
        summary_line = f"Progress: {jobs_done}/{self.total_jobs} [{percentage:3d}%] {progress_bar_str} | ETA: {eta_str}"
        print(f"\033[2K{summary_line}".ljust(terminal_width))
