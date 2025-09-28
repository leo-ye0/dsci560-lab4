import schedule
import time
import sys
import setup_database 
import reddit_scraper
import clusters_use

def automate():
    params = sys.argv[1:]
    hours, minutes, seconds = 0, 0, 0

    for param in params:
        if 'h' in param:
            num = param.split('h')[0]
            num = int(num)
            hours += num
        elif 'm' in param:
            num = param.split('m')[0]
            num = int(num)
            minutes += num
        elif 's' in param:
            num = param.split('s')[0]
            num = int(num)
            seconds += num
    
    time_scheduled = hours * 60 * 60 + minutes * 60 + seconds

    print('step 1')
    schedule.every(time_scheduled).seconds.do(setup_database.setup_database())
    print('step ')
    schedule.every(time_scheduled).seconds.do(reddit_scraper.main())
    print('step 3')
    schedule.every(time_scheduled).seconds.do(clusters_use.main())

if __name__ == '__main__':
    automate()