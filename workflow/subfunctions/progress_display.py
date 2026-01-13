import os
import time
import sys
import threading
import signal
import concurrent.futures
import glob

class ProgressDisplay:
    """
    A class to display progress for multiprocessing tasks in the terminal.
    """
    def __init__(self, jobs, num_processes, log_dir=None, cleanup_globs=None):
        self.jobs = jobs
        self.num_processes = num_processes
        self.total_jobs = len(jobs)
        self.statuses = {i: "pending" for i in range(self.total_jobs)}
        self.results = []
        self.start_time = None
        self.next_job_to_run = 0
        self.last_lines_printed = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ticker_thread = None
        self.avg_time_per_job = 0
        
        # Logging and Signal handling state
        self.log_dir = log_dir
        self.cleanup_globs = cleanup_globs or []
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.original_stdout_obj = None
        self.original_stderr_obj = None
        self.log_file = None
        self.terminal_out = None
        self.original_sigint_handler = None

    def start(self):
        """
        Initializes the display and starts the first batch of jobs.
        """
        # Setup logging if log_dir is provided
        if self.log_dir:
            self._setup_logging()
        else:
            self.terminal_out = sys.stdout

        # Setup signal handler for Ctrl+C
        self._setup_signal_handler()

        self.start_time = time.time()
        for i in range(min(self.num_processes, self.total_jobs)):
            self.statuses[i] = "running"
            self.next_job_to_run += 1
        
        if self.total_jobs > 0:
            self._redraw()
            self.ticker_thread = threading.Thread(target=self._tick)
            self.ticker_thread.start()

    def _setup_logging(self):
        log_file_path = os.path.join(self.log_dir, 'segmentation_log.txt')
        
        # Save original file descriptors
        self.original_stdout_fd = os.dup(1)
        self.original_stderr_fd = os.dup(2)
        
        # Open log file
        self.log_file = open(log_file_path, 'w')
        
        # Create a file object for the terminal output
        self.terminal_out = os.fdopen(self.original_stdout_fd, 'w')

        # Redirect stdout/stderr to log file at FD level
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.log_file.fileno(), 1)
        os.dup2(self.log_file.fileno(), 2)
        
        # Update Python objects
        self.original_stdout_obj = sys.stdout
        self.original_stderr_obj = sys.stderr
        sys.stdout = self.log_file
        sys.stderr = self.log_file

    def _teardown_logging(self):
        if self.log_dir and self.log_file:
            try:
                # Restore Python objects
                if self.original_stdout_obj:
                    sys.stdout = self.original_stdout_obj
                if self.original_stderr_obj:
                    sys.stderr = self.original_stderr_obj

                # Restore FDs
                if self.original_stdout_fd is not None:
                    os.dup2(self.original_stdout_fd, 1)
                if self.original_stderr_fd is not None:
                    os.dup2(self.original_stderr_fd, 2)
                
                # Close the duplicate FDs/wrappers
                if self.log_file:
                    try: self.log_file.close()
                    except: pass
                    self.log_file = None
                
                if self.terminal_out:
                    try: self.terminal_out.close()
                    except: pass
                    self.terminal_out = None
                    self.original_stdout_fd = None
                
                if self.original_stderr_fd is not None:
                    try: os.close(self.original_stderr_fd)
                    except: pass
                    self.original_stderr_fd = None
            except Exception:
                pass 

    def _setup_signal_handler(self):
        try:
            self.original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
            signal.signal(signal.SIGTERM, self._handle_sigint)
        except (ValueError, RuntimeError):
            pass

    def _teardown_signal_handler(self):
        try:
            if self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except (ValueError, RuntimeError):
            pass

    def _handle_sigint(self, signum, frame):
        # Immediate exit is safest. Use os.write to bypass Python buffering.
        target_fd = self.original_stdout_fd if self.original_stdout_fd else 1
        try:
            os.write(target_fd, b"\n\n\033[91mProcess interrupted by user! Stopping...\033[0m\n")
        except:
            pass
        
        # Cleanup temporary files
        if self.log_dir and self.cleanup_globs:
            for pattern in self.cleanup_globs:
                search_pattern = os.path.join(self.log_dir, pattern)
                for f in glob.glob(search_pattern):
                    try:
                        os.remove(f)
                    except:
                        pass

        self._teardown_logging()
        os._exit(1)

    def wait_for_futures(self, futures):
        """
        Waits for futures to complete in an interruptible way.
        Returns a list of results.
        """
        all_results = []
        not_done = set(futures)
        while not_done:
            try:
                # Wait for at least one future to complete, or timeout after 0.5s
                done, not_done = concurrent.futures.wait(
                    not_done, 
                    return_when=concurrent.futures.FIRST_COMPLETED, 
                    timeout=0.5
                )
                for future in done:
                    result = future.result()
                    all_results.append(result)
                    self.update(result)
            except (KeyboardInterrupt, SystemExit):
                self._handle_sigint(None, None)
            except Exception as e:
                if self.terminal_out:
                    self.terminal_out.write(f"\nError in main loop: {e}\n")
                break
        return all_results

    def wait_for_pool(self, pool, async_results):
        """
        Waits for multiprocessing pool results in an interruptible way.
        """
        all_results = []
        pending = list(async_results)
        while pending:
            try:
                # Check each pending result with a short timeout
                still_pending = []
                for r in pending:
                    if r.ready():
                        result = r.get()
                        all_results.append(result)
                        self.update(result)
                    else:
                        still_pending.append(r)
                pending = still_pending
                if pending:
                    time.sleep(0.5) # Sleep to avoid busy waiting
            except (KeyboardInterrupt, SystemExit):
                self._handle_sigint(None, None)
            except Exception as e:
                if self.terminal_out:
                    self.terminal_out.write(f"\nError in pool loop: {e}\n")
                break
        return all_results

    def _tick(self):
        """
        Periodically updates the display to refresh the ETA.
        """
        while not self.stop_event.is_set():
            time.sleep(1)
            self._redraw()

    def update(self, result):
        """
        Updates the status of a job and redraws the display.
        """
        with self.lock:
            if result:
                index = result['index']
                error = result.get('error')
                
                if error == 'skipped':
                    self.statuses[index] = "skipped"
                elif error:
                    self.statuses[index] = "failed"
                else:
                    self.statuses[index] = "done"
                
                self.results.append(result)
                
                # Update average time per job
                # We use all finished jobs (including skipped) to estimate the rate of progress
                finished_count = len(self.results)
                elapsed_time = time.time() - self.start_time
                self.avg_time_per_job = elapsed_time / finished_count

                if self.next_job_to_run < self.total_jobs:
                    self.statuses[self.next_job_to_run] = "running"
                    self.next_job_to_run += 1
        
        self._redraw()

    def finish(self):
        """
        Cleans up the display by moving the cursor to the end.
        """
        self.stop_event.set()
        if self.ticker_thread:
            self.ticker_thread.join()
        
        self._teardown_signal_handler()
        if self.log_dir:
            self._teardown_logging()

    def _redraw(self):
        """
        Redraws the entire progress display.
        """
        with self.lock:
            if not self.terminal_out:
                return
            try:
                terminal_height, terminal_width = os.get_terminal_size()
            except OSError:
                terminal_height, terminal_width = 24, 80

            output_buffer = []

            # Move cursor up to overwrite previous output
            if self.last_lines_printed > 0:
                output_buffer.append(f"\033[{self.last_lines_printed}A")

            # Clear from cursor down to ensure clean slate
            output_buffer.append("\033[0J")

            lines = []

            # Redraw active jobs section
            active_jobs = [self.jobs[i] for i, s in self.statuses.items() if s == "running"]
            header = f"--- Active Jobs ({len(active_jobs)}/{self.num_processes}) [Ctrl+C to Stop] ---"
            lines.append(header[:terminal_width-1])

            #-3 for header, summary, and potential message line
            max_visible_jobs = min(self.num_processes, terminal_height - 3) 
            
            # Determine how many lines to use for jobs
            lines_for_jobs = max_visible_jobs

            if len(active_jobs) > max_visible_jobs:
                # We need one line for "more jobs" message
                jobs_to_show = max_visible_jobs - 1
                for i in range(jobs_to_show):
                    line = f"  {active_jobs[i]}"
                    lines.append(line[:terminal_width-1])
                
                remaining_jobs = len(active_jobs) - jobs_to_show
                msg = f"--- and {remaining_jobs} more active jobs ---"
                lines.append(msg[:terminal_width-1])
            else:
                # Show all active jobs
                for job in active_jobs:
                    line = f"  {job}"
                    lines.append(line[:terminal_width-1])
                
                # Pad with blank lines to maintain stable height
                padding_needed = max_visible_jobs - len(active_jobs)
                for _ in range(padding_needed):
                    lines.append("")


            # Redraw summary
            finished_jobs = len(self.results)
            skipped_jobs = sum(1 for s in self.statuses.values() if s == "skipped")
            failed_jobs = sum(1 for s in self.statuses.values() if s == "failed")
            done_jobs = sum(1 for s in self.statuses.values() if s == "done")
            
            if self.total_jobs > 0:
                progress_fraction = finished_jobs / self.total_jobs
                percentage = int(progress_fraction * 100)

                # Dynamic bar width based on terminal size, min 10, max 40
                reserved_chars = 60 
                bar_width = max(10, min(40, terminal_width - reserved_chars))
                
                filled_length = int(bar_width * progress_fraction)
                bar = '█' * filled_length + '░' * (bar_width - filled_length)
                progress_bar_str = f"\033[92m{bar}\033[0m"
            else:
                percentage = 0
                progress_bar_str = ""

            if finished_jobs > 0:
                current_elapsed_time = time.time() - self.start_time
                total_estimated_time = self.avg_time_per_job * self.total_jobs
                eta_seconds = int(total_estimated_time - current_elapsed_time)
                
                if eta_seconds < 0:
                    eta_seconds = 0
                    
                eta_str = f"{eta_seconds // 60:02d}m{eta_seconds % 60:02d}s"
            else:
                eta_str = "calculating..."
            
            # Build summary line with skipped count if any
            summary_parts = [f"Progress: {finished_jobs}/{self.total_jobs} [{percentage:3d}%] {progress_bar_str}"]
            if skipped_jobs > 0:
                summary_parts.append(f"| \033[94mAlready Computed: {skipped_jobs}\033[0m")
            if failed_jobs > 0:
                summary_parts.append(f"| \033[91mFailed: {failed_jobs}\033[0m")
            summary_parts.append(f"| ETA: {eta_str}")
            
            summary_line = " ".join(summary_parts)
            lines.append(summary_line[:terminal_width+20]) # Allow some extra for ANSI codes
            
            # Add newlines to lines
            final_lines = [l + "\n" for l in lines]
            
            output_buffer.extend(final_lines)
            
            self.terminal_out.write("".join(output_buffer))
            self.terminal_out.flush()
            
            self.last_lines_printed = len(final_lines)
