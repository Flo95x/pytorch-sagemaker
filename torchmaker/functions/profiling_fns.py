import contextlib
import time


@contextlib.contextmanager
def log_time(fn_name, is_on=True, log_start=True, custom_logger=None):
    if is_on:
        if log_start:
            if custom_logger:
                custom_logger.info("Processing: " + fn_name)
            else:
                print("Processing: " + fn_name)
        start = time.time()
        yield
        end = time.time()
        msg = "Processing: " + fn_name + ' finished in ' + str(end - start) + 's'
        if custom_logger:
            custom_logger.info(msg)
        else:
            print(msg)
    else:
        pass
        yield
        pass