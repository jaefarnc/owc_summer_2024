""" 
//TERMINAL CODE:

//START A VIRTUAL ENVIRONMENT
pip install virtualenv
python -m venv venv
python venv/scripts/Activate

//INSTALL PACKAGES
pip install pytest-playwright
pip install bs4
playwright install
pip install playwright-stealth

//GO INTO THE FOLDER WHERE test.py IS
mkdir niter
cd nitter

//RUN THE CODE
//FIRST COPY THE test.py FILE INTO NITTER FOLDER THAT YOU JUST MADE
python test.py

 """

import csv
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

from bs4 import BeautifulSoup
import time
import random


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    stealth_sync(page)
    #Wait until some element has loaded
    try:
        page.goto("http://192.168.24.1:1000/login?35ae612493473f02")
        time.sleep(10)
        page.fill("input[name=username]", "mohammed_b220404CS")
        page.fill("input[name = password]", "B220404CS")
        button = page.locator('input[value = Continue]')
        button.click()
        while(True):
            time.sleep(10)
            page.reload()
    except:
        print("Connection Failed")
        time.sleep(100)
    time.sleep(10)
    # button = page.locator('//html/body/div/div/form/p[2]/a')
    # button.click()
    time.sleep(10)



#for each profile:
    # get username
    # get tweet count
    # get joining date
    # get following count
    # get followers
    # get likes
    # click on tweets
        #   get tweet text
    # click on replies
        # get reply text

#store it as a dictionary, all_profile_details