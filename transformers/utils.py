from datetime import datetime
import os

# find newest model from results dir based on dir naming pattern
def get_latest_trainingrun(results_dir):
    for (_, dirs, _) in os.walk(results_dir):
        break

    date_pattern = "%m-%d-%Y"
    max_date = datetime.strptime("01-01-1970", date_pattern)
    dir: str
    for dir in dirs:
        try:
            dir_date = datetime.strptime(dir, date_pattern)
            if  dir_date > max_date:
                max_date = dir_date
        except:
            print(f"can't parse date {dir}")
    return max_date.strftime(date_pattern)

# get latest checkpoint in model dir
def get_latest_checkpoint(model_dir):
    print(f"trying to get checkpoints in {model_dir}")
    for (_, dirs, _) in os.walk(model_dir):
        break
    checkpoints = [dir for dir in dirs if dir.startswith("checkpoint")]
    last_cp = 0
    for checkpoint in checkpoints:
        _, curr_cp = checkpoint.split("-")
        curr_cp = int(curr_cp)
        if curr_cp > last_cp:
            last_cp = curr_cp
    return f"checkpoint-{last_cp}"