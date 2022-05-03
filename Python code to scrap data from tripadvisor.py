# %%
import sys
import csv
from selenium import webdriver
import time

# default path to file to store data
path_to_file = "/Users/Dell/Documents/a/b/c/org.csv"

# default number of scraped pages
num_page = 50

# default tripadvisor website of hotel or things to do (attraction/monument) 
url = "https://www.tripadvisor.com/Attraction_Review-g60491-d189546-Reviews-Jackson_Hole_Rodeo-Jackson_Jackson_Hole_Wyoming.html"
#url = "https://www.tripadvisor.com/Attraction_Review-g187791-d192285-Reviews-Colosseum-Rome_Lazio.html"

# if you pass the inputs in the command line
if (len(sys.argv) == 4):
    path_to_file = sys.argv[1]
    num_page = int(sys.argv[2])
    url = sys.argv[3]

# import the webdriver
driver = webdriver.Chrome('chromedriver.exe')
driver.get(url)

# open the file to save the review
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)

# change the value inside the range to save more or less reviews
for i in range(0, num_page):

    # expand the review 
    time.sleep(5)
    #driver.find_element_by_xpath("//button[@class='bfQwA _G B- _S _T c G_ P0 ddFHE cnvzr']").click()

    container = driver.find_elements_by_xpath(".//div[@class='ffbzW _c']")
    #dates = driver.find_elements_by_xpath(".//div[@class='_2fxQ4TOx']")
    
# 'eRduX'
    #driver.find_elements_by_xpath('.//div[@class="ldSaR.t._U.Za"]').click()  #Pop_up

    for j in range(len(container)):


        review = container[j].find_element_by_xpath(".//span[@class='NejBf']").text.replace("\n", "  ")
        
        date = container[j].find_element_by_xpath(".//div[@class='eRduX']").text.replace("\n", "  ")

        tagline = container[j].find_element_by_xpath(".//div[@class='WlYyy cPsXC bLFSo cspKb dTqpp']").text.replace("\n", "  ")


        location = container[j].find_element_by_xpath(".//div[@class='WlYyy diXIH bQCoY']").text.replace("\n", "  ")
        
        rev = container[j].find_element_by_xpath(".//div[@class='WlYyy diXIH dDKKM']").text.replace("\n", "  ")

    
        csvWriter.writerow([date,location,tagline,rev]) 

        #time.sleep(10)
        
    # change the page            
    driver.find_element_by_xpath('.//div[@class="cCnaz"]').click()

driver.quit()

# %%
