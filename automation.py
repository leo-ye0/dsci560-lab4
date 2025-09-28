import schedule
import time
import sys
import setup_database 
import reddit_scraper
import clusters_use
import subprocess

def automate():
    params = sys.argv[1:]
    hours, minutes, seconds = 0, 0, 0
    param = params[0]
    num_post = params[1]

    num_str = ''.join(filter(str.isdigit, param))

    num = int(num_str)

    if 'h' in param:
        hours += num
    elif 'm' in param:
        minutes += num
    elif 's' in param:
        seconds += num
    
    time_scheduled = hours * 60 * 60 + minutes * 60 + seconds

    count = 0

    def job():
        nonlocal count
        count += 1
        print(f'RUN {count}')
        print('\nRunning task 1: setup_database.py')
        setup_database.setup_database()
        print('\nRunning task 2: reddit_scraper.py')
        subprocess.run(['python3', 'reddit_scraper.py', str(num_post)])
        print('\nRunning task 3: clusters_use.py')
        clusters_use.main()
        print(f'\nFINISHED RUN {count}')
        print(f'---\n')

    schedule.every(time_scheduled).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    automate()