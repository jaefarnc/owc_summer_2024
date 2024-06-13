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
NUM_PAGES_PER_PROFILE=5
nitterurl = "https://nitter.poast.org"
searchurl = nitterurl +  "/search?f=tweets&q="

queries = [
    # Tarantula products
    "Live tarantulas", "Tarantula spiders", "Tarantula venom", "Spider extract", "Tarantula silk",
    "Live spiders", "Spider webs", "Tarantula fangs", "Spider eggs", "Tarantula legs",
    "Spider body", "Tarantula skins", "Spider carapace", "Tarantula claws", "Live arachnids",
    "Tarantula carapace", "Spider toxin", "Tarantula moltings", "Spider legs", "Live arachnids",
    "Tarantula shells", "Spider exoskeleton", "Tarantula hatchlings", "Spiderlings", "Tarantula eggs",
    "Spider molts", "Tarantula toxin", "Spider venom", "Tarantula webs", "Live tarantula pets",
    "Spider husks", "Tarantula specimens", "Spider claws", "Tarantula skin", "Spider hatchlings",
    "Tarantula shells", "Spider exoskeletons", "Tarantula body parts", "Spider parts", "Tarantula molts",
    "Spider claws", "Live tarantulas", "Spider fragments", "Tarantula husks", "Spider bodies",
    "Tarantula fragments", "Spider moltings", "Tarantula carapaces", "Spider carapace", "Live spider pets",
    "Tarantula toxin", "Spider fangs", "Tarantula legs", "Spider skins", "Tarantula exoskeleton",
    "Spider shells", "Tarantula webs", "Spider molts", "Tarantula molting", "Spider fragments",
    "Live arachnids", "Spider carapaces", "Tarantula spiderlings", "Spider moltings", "Tarantula specimens",
    "Spider bodies", "Tarantula eggs", "Spider exoskeletons", "Tarantula shells", "Spider skin",
    "Live spiders", "Spider hatchlings", "Tarantula claws", "Spider moltings", "Tarantula exoskeleton",
    "Spider eggs", "Tarantula fragments", "Spider claws", "Live tarantula", "Spider silk",
    "Tarantula bodies", "Spider fragments", "Tarantula husks", "Spider carapace", "Live arachnids",
    "Spider body", "Tarantula legs", "Spider molts", "Tarantula molts", "Spider specimens",
    "Tarantula exoskeletons", "Spider skins", "Tarantula toxin", "Spider venom", "Tarantula webs",
    "Spider carapaces", "Tarantula silk", "Spider fragments", "Live spider", "Tarantula parts",
    "Spider fangs", "Tarantula skin", "Spider molts", "Tarantula carapaces", "Spider husks",
    "Tarantula venom", "Spider fragments", "Live tarantulas", "Tarantula body", "Spider bodies",
    "Tarantula fragments", "Spider molting", "Tarantula claws", "Spider shells", "Tarantula eggs",
    "Spider eggs", "Tarantula body parts", "Spider fragments", "Tarantula exoskeletons", "Spider skins",
    "Tarantula husks", "Spider molting", "Live tarantula pets", "Spider carapaces", "Tarantula fragments",
    "Spider fragments", "Tarantula skin", "Spider fangs", "Live tarantula", "Spider webs",
    "Tarantula shells", "Spider body", "Tarantula parts", "Spider molts", "Live spider pets",
    "Tarantula bodies", "Spider fragments", "Tarantula molts", "Spider claws", "Tarantula silk",
    "Spider exoskeleton", "Tarantula specimens", "Spider venom", "Live tarantulas", "Spider fragments",
    "Tarantula body", "Spider molting", "Tarantula eggs", "Spider bodies",

    # Varanus products
    "Live monitor lizards", "Monitor lizard skins", "Monitor lizard meat", "Varanus skins", "Monitor lizard claws",
    "Varanus meat", "Monitor lizard eggs", "Varanus claws", "Monitor lizard hides", "Varanus hides",
    "Monitor lizard parts", "Varanus parts", "Monitor lizard bones", "Varanus bones", "Monitor lizard extracts",
    "Varanus extracts", "Live varanus", "Monitor lizard scales", "Varanus scales", "Monitor lizard oil",
    "Varanus oil", "Monitor lizard leather", "Varanus leather", "Monitor lizard organs", "Varanus organs",
    "Monitor lizard fat", "Varanus fat", "Monitor lizard blood", "Varanus blood", "Monitor lizard skin",
    "Varanus skin", "Monitor lizard shells", "Varanus shells", "Monitor lizard fragments", "Varanus fragments",
    "Monitor lizard venom", "Varanus venom", "Monitor lizard hides", "Varanus hides", "Monitor lizard gallbladders",
    "Varanus gallbladders", "Monitor lizard tongues", "Varanus tongues", "Monitor lizard embryos", "Varanus embryos",
    "Monitor lizard intestines", "Varanus intestines", "Monitor lizard sinews", "Varanus sinews", "Monitor lizard cartilage",
    "Varanus cartilage", "Monitor lizard tendons", "Varanus tendons", "Monitor lizard marrow", "Varanus marrow",
    "Monitor lizard spinal fluid", "Varanus spinal fluid", "Monitor lizard ligaments", "Varanus ligaments",
    "Monitor lizard guts", "Varanus guts", "Monitor lizard muscles", "Varanus muscles", "Monitor lizard bile",
    "Varanus bile", "Monitor lizard bones", "Varanus bones", "Monitor lizard fluids", "Varanus fluids",
    "Monitor lizard brains", "Varanus brains", "Monitor lizard claws", "Varanus claws", "Monitor lizard talons",
    "Varanus talons", "Monitor lizard fangs", "Varanus fangs", "Monitor lizard skulls", "Varanus skulls",
    "Monitor lizard vertebrae", "Varanus vertebrae", "Monitor lizard ribs", "Varanus ribs", "Monitor lizard spines",
    "Varanus spines", "Monitor lizard tails", "Varanus tails", "Monitor lizard paws", "Varanus paws",
    "Monitor lizard jaws", "Varanus jaws", "Monitor lizard jaws", "Varanus jaws", "Monitor lizard noses",
    "Varanus noses", "Monitor lizard eyes", "Varanus eyes", "Monitor lizard ears", "Varanus ears",
    "Monitor lizard skins", "Varanus skins", "Monitor lizard hearts", "Varanus hearts", "Monitor lizard livers",
    "Varanus livers", "Monitor lizard kidneys", "Varanus kidneys", "Monitor lizard spleens", "Varanus spleens",
    "Monitor lizard pancreas", "Varanus pancreas", "Monitor lizard reproductive organs", "Varanus reproductive organs",
    "Monitor lizard stomachs", "Varanus stomachs", "Monitor lizard colons", "Varanus colons", "Monitor lizard rectums",
    "Varanus rectums", "Monitor lizard esophagus", "Varanus esophagus", "Monitor lizard throats", "Varanus throats",
    "Monitor lizard windpipes", "Varanus windpipes", "Monitor lizard tracheas", "Varanus tracheas", "Monitor lizard gizzards",
    "Varanus gizzards", "Monitor lizard intestines", "Varanus intestines", "Monitor lizard testicles", "Varanus testicles",

    # Orchid products
    "Orchid plants", "Orchid roots", "Orchid leaves", "Orchid flowers", "Orchid seeds",
    "Wild orchids", "Orchid bulbs", "Orchid stems", "Orchid cuttings", "Orchid petals",
    "Orchid extracts", "Orchid oils", "Orchid powders", "Orchid tinctures", "Orchid roots",
    "Orchid rhizomes", "Orchid tubers", "Orchid blossoms", "Orchid capsules", "Orchid sap",
    "Orchid pollen", "Orchid nectar", "Orchid resins", "Orchid gums", "Orchid gels",
    "Orchid spikes", "Orchid buds", "Orchid sprigs", "Orchid shoots", "Orchid infusions",
    "Orchid tonics", "Orchid syrups", "Orchid solutions", "Orchid concentrates", "Orchid elixirs",
    "Orchid extracts", "Orchid essences", "Orchid oils", "Orchid balms", "Orchid creams",
    "Orchid lotions", "Orchid salves", "Orchid ointments", "Orchid pastes", "Orchid gels",
    "Orchid serums", "Orchid gels", "Orchid sprays", "Orchid powders", "Orchid tinctures",
    "Orchid capsules", "Orchid tablets", "Orchid pills", "Orchid powders", "Orchid flakes",
    "Orchid chips", "Orchid pieces", "Orchid shards", "Orchid fragments", "Orchid scraps",
    "Orchid remnants", "Orchid residues", "Orchid debris", "Orchid remnants", "Orchid traces",
    "Orchid leftovers", "Orchid remains", "Orchid extracts", "Orchid distillates", "Orchid concentrates",
    "Orchid solutions", "Orchid emulsions", "Orchid infusions", "Orchid decoctions", "Orchid distillates",
    "Orchid reductions", "Orchid syrups", "Orchid nectars", "Orchid saps", "Orchid gums",
    "Orchid resins", "Orchid balms", "Orchid ointments", "Orchid salves", "Orchid creams",
    "Orchid pastes", "Orchid powders", "Orchid tinctures", "Orchid capsules", "Orchid tablets",
    "Orchid pills", "Orchid syrups", "Orchid extracts", "Orchid essences", "Orchid oils",
    "Orchid balsams", "Orchid unguents", "Orchid decoctions", "Orchid infusions", "Orchid emulsions",
    "Orchid distillates", "Orchid solutions", "Orchid syrups", "Orchid nectars", "Orchid saps",
    "Orchid gums", "Orchid resins", "Orchid creams", "Orchid pastes", "Orchid powders",
    "Orchid tinctures", "Orchid capsules", "Orchid tablets", "Orchid pills", "Orchid powders",
    "Orchid flakes", "Orchid chips", "Orchid pieces", "Orchid shards", "Orchid fragments",
    "Orchid scraps", "Orchid remnants", "Orchid residues", "Orchid debris", "Orchid remnants",
    "Orchid traces", "Orchid leftovers", "Orchid remains", "Orchid powders", "Orchid extracts",
    "Orchid oils", "Orchid balms", "Orchid creams", "Orchid lotions", "Orchid salves",
    "Orchid ointments", "Orchid pastes", "Orchid gels", "Orchid serums", "Orchid sprays",

    # Rosewood products
    "Rosewood logs", "Rosewood timber", "Rosewood planks", "Rosewood slabs", "Rosewood boards",
    "Rosewood veneers", "Rosewood sheets", "Rosewood panels", "Rosewood strips", "Rosewood pieces",
    "Rosewood chips", "Rosewood fragments", "Rosewood scraps", "Rosewood remnants", "Rosewood residues",
    "Rosewood debris", "Rosewood leftovers", "Rosewood remains", "Rosewood dust", "Rosewood sawdust",
    "Rosewood shavings", "Rosewood splinters", "Rosewood filings", "Rosewood powder", "Rosewood pellets",
    "Rosewood briquettes", "Rosewood blocks", "Rosewood cubes", "Rosewood chunks", "Rosewood fragments",
    "Rosewood pieces", "Rosewood slats", "Rosewood battens", "Rosewood strips", "Rosewood timbers",
    "Rosewood laths", "Rosewood boards", "Rosewood sheets", "Rosewood panels", "Rosewood slabs",
    "Rosewood planks", "Rosewood logs", "Rosewood billets", "Rosewood beams", "Rosewood girders",
    "Rosewood spars", "Rosewood poles", "Rosewood staves", "Rosewood rods", "Rosewood stakes",
    "Rosewood pegs", "Rosewood dowels", "Rosewood pins", "Rosewood pieces", "Rosewood blocks",
    "Rosewood chips", "Rosewood fragments", "Rosewood scraps", "Rosewood remnants", "Rosewood residues",
    "Rosewood debris", "Rosewood leftovers", "Rosewood remains", "Rosewood dust", "Rosewood shavings",
    "Rosewood splinters", "Rosewood filings", "Rosewood powder", "Rosewood pellets", "Rosewood briquettes",
    "Rosewood logs", "Rosewood timber", "Rosewood planks", "Rosewood slabs", "Rosewood boards",
    "Rosewood veneers", "Rosewood sheets", "Rosewood panels", "Rosewood strips", "Rosewood pieces",
    "Rosewood chips", "Rosewood fragments", "Rosewood scraps", "Rosewood remnants", "Rosewood residues",
    "Rosewood debris", "Rosewood leftovers", "Rosewood remains", "Rosewood dust", "Rosewood sawdust",
    "Rosewood shavings", "Rosewood splinters", "Rosewood filings", "Rosewood powder", "Rosewood pellets",
    "Rosewood briquettes", "Rosewood blocks", "Rosewood cubes", "Rosewood chunks", "Rosewood fragments",
    "Rosewood pieces", "Rosewood slats", "Rosewood battens", "Rosewood strips", "Rosewood timbers",
    "Rosewood laths", "Rosewood boards", "Rosewood sheets", "Rosewood panels", "Rosewood slabs",
    "Rosewood planks", "Rosewood logs", "Rosewood billets", "Rosewood beams", "Rosewood girders",
    "Rosewood spars", "Rosewood poles", "Rosewood staves", "Rosewood rods", "Rosewood stakes",
    "Rosewood pegs", "Rosewood dowels", "Rosewood pins", "Rosewood pieces", "Rosewood blocks",
    "Rosewood chips", "Rosewood fragments", "Rosewood scraps", "Rosewood remnants", "Rosewood residues",
    "Rosewood debris", "Rosewood leftovers", "Rosewood remains", "Rosewood dust", "Rosewood shavings",

    # Big cat products (Panthera spp.)
    "Tiger bones", "Lion bones", "Leopard bones", "Panther bones", "Jaguar bones",
    "Tiger claws", "Lion claws", "Leopard claws", "Panther claws", "Jaguar claws",
    "Tiger skins", "Lion skins", "Leopard skins", "Panther skins", "Jaguar skins",
    "Tiger teeth", "Lion teeth", "Leopard teeth", "Panther teeth", "Jaguar teeth",
    "Tiger whiskers", "Lion whiskers", "Leopard whiskers", "Panther whiskers", "Jaguar whiskers",
    "Tiger fangs", "Lion fangs", "Leopard fangs", "Panther fangs", "Jaguar fangs",
    "Tiger fur", "Lion fur", "Leopard fur", "Panther fur", "Jaguar fur",
    "Tiger hides", "Lion hides", "Leopard hides", "Panther hides", "Jaguar hides",
    "Tiger bones", "Lion bones", "Leopard bones", "Panther bones", "Jaguar bones",
    "Tiger organs", "Lion organs", "Leopard organs", "Panther organs", "Jaguar organs",
    "Tiger blood", "Lion blood", "Leopard blood", "Panther blood", "Jaguar blood",
    "Tiger bile", "Lion bile", "Leopard bile", "Panther bile", "Jaguar bile",
    "Tiger fats", "Lion fats", "Leopard fats", "Panther fats", "Jaguar fats",
    "Tiger oils", "Lion oils", "Leopard oils", "Panther oils", "Jaguar oils",
    "Tiger extracts", "Lion extracts", "Leopard extracts", "Panther extracts", "Jaguar extracts",
    "Tiger glands", "Lion glands", "Leopard glands", "Panther glands", "Jaguar glands",
    "Tiger sinews", "Lion sinews", "Leopard sinews", "Panther sinews", "Jaguar sinews",
    "Tiger tendons", "Lion tendons", "Leopard tendons", "Panther tendons", "Jaguar tendons",
    "Tiger cartilage", "Lion cartilage", "Leopard cartilage", "Panther cartilage", "Jaguar cartilage",
    "Tiger ligaments", "Lion ligaments", "Leopard ligaments", "Panther ligaments", "Jaguar ligaments",
    "Tiger marrow", "Lion marrow", "Leopard marrow", "Panther marrow", "Jaguar marrow",
    "Tiger fluids", "Lion fluids", "Leopard fluids", "Panther fluids", "Jaguar fluids",
    "Tiger spinal fluid", "Lion spinal fluid", "Leopard spinal fluid", "Panther spinal fluid", "Jaguar spinal fluid",
    "Tiger brains", "Lion brains", "Leopard brains", "Panther brains", "Jaguar brains",
    "Tiger paws", "Lion paws", "Leopard paws", "Panther paws", "Jaguar paws",
    "Tiger jaws", "Lion jaws", "Leopard jaws", "Panther jaws", "Jaguar jaws",
    "Tiger noses", "Lion noses", "Leopard noses", "Panther noses", "Jaguar noses",
    "Tiger eyes", "Lion eyes", "Leopard eyes", "Panther eyes", "Jaguar eyes",
    "Tiger ears", "Lion ears", "Leopard ears", "Panther ears", "Jaguar ears",
    "Tiger hearts", "Lion hearts", "Leopard hearts", "Panther hearts", "Jaguar hearts",
    "Tiger livers", "Lion livers", "Leopard livers", "Panther livers", "Jaguar livers",
    "Tiger kidneys", "Lion kidneys", "Leopard kidneys", "Panther kidneys", "Jaguar"]

profile_num=1
"""BS4 OPERATIONS TO GET PROFILES via KEYWORD QUERY"""
def get_soup(page):
    html = page.inner_html('body')
    return  BeautifulSoup(html,'html.parser')

def extract_profiles(soup):
    global profile_num
    profiles = []
    profile_links = soup.find_all('a',class_='username')
    for link in profile_links:
        profiles.append(nitterurl + link['href'])
        print(profile_num," ",nitterurl+link['href'])
        profile_num+=1
    return profiles


def get_profiles_from_query(page, query):
    profiles = []
    search_url = searchurl + query.replace(" ","+")

    try:
        page.goto(search_url)
    except:
        return profiles    

    for count in range(1,20):
        soup = get_soup(page)
        search_url =page.url

        profiles.extend(extract_profiles(soup))
        #next page
        try:
            rtag = page.locator('//a[text()="Load more"]')
            rtag.wait_for(state="visible",timeout=5000)
            rtag.click()
        except:
            return profiles      
        delay = random.randint(1,5)
        time.sleep(delay)

    return profiles

def get_posts(soup):
    tweets =[]
    replies =[]
    try:
        posts = soup.find_all('div', class_ = 'timeline-item')
    except:
        return {}

    delay = random.randint(1,5)
    time.sleep(delay)

    for post in posts:
        try:
            tweet_link = post.find('a', recursive = False)['href']
            tweet_date = post.find('span',class_="tweet-date").find('a')['title'].strip()
            
            #skip if retweet
            isRetweet = bool(post.find('div',class_="retweet-header"))
            if(isRetweet):
                continue

            isReply = bool(post.find('div',class_="replying-to"))
            isRetweet = bool(post.find('div',class_="retweet-header"))

            tweet_content = post.find('div',class_="tweet-content media-body").get_text(strip=True)
            content={
                'tweet_link':tweet_link,
                'tweet_date':tweet_date,
                'tweet_content':tweet_content
            }
            if(isReply):
                r_tag = post.find('div',class_='replying-to')
                a_tag = r_tag.find('a',recursive = False)
                content['replying-to']=a_tag.get_text(strip=True)
            

            b_tag = post.find('div',class_="tweet-body")
            ts_tag = b_tag.find('div',class_="tweet-stats",recursive=False)

            stats = ["comments","retweets","quotes","likes"]
            s_tags = ts_tag.find_all('div',class_='icon-container')[0:4]

            for i in range(0,4):
                s_tag = s_tags[i]
                content[stats[i]]=s_tag.get_text(strip=True)

            if(isReply):
                replies.append(content)
            else:
                tweets.append(content)
        except:
            continue

    return {
        'tweets':tweets,
        'replies':replies
    }


def extract_profile_details(page, profile_link):

    try:
        u_tag = page.locator('//a[@class="profile-card-fullname"]')
        username = u_tag.get_attribute('title')
    
        h_tag = page.locator('//a[@class="profile-card-username"]')
        handle = h_tag.inner_text()
        #joining date  
        # Locate the div tag with the class "profile-joindate"
        # Locate the span element using its XPath
        j_tag = page.locator('//div[@class="profile-joindate"]/span[1]')
        join_date = j_tag.get_attribute('title')

        #stat list
        #tweet_count
        s_tag =page.locator('//li[@class="posts"]/span[2]')
        tweet_count = s_tag.inner_text().strip()

        #following
        s_tag =page.locator('//li[@class="following"]/span[2]')
        following_count = s_tag.inner_text().strip()

        #followers
        s_tag =page.locator('//li[@class="followers"]/span[2]')
        followers_count = s_tag.inner_text().strip()

        #likes
        s_tag =page.locator('//li[@class="likes"]/span[2]')
        likes_count = s_tag.inner_text().strip()
    
        posts ={"tweets":[], "replies":[]}
        for i in range(0,NUM_PAGES_PER_PROFILE):
            try:
                html = page.inner_html('div.timeline')
                soup = BeautifulSoup(html,'html.parser')
                t_posts = get_posts(soup)
                if 'tweets' not in t_posts:
                    break
                posts['tweets'].extend(t_posts['tweets'])
                posts['replies'].extend(t_posts['replies'])
            
                lm_tag = page.locator('//a[text()="Load more"]')
                lm_tag.wait_for(state='visible',timeout=5000)
                lm_tag.click()
            except:
                break
        #get the first tweet
        return {
            'username':username,
            'handle':handle,
            'join_date': join_date,
            'tweet_count':tweet_count,
            'following_count':following_count,
            'followers_count':followers_count,
            'likes_count':likes_count,
            'tweets':posts['tweets'],
            'replies':posts['replies']
        }
    except:
        print("Not Found")
        return {}
    
csv_file = 'output.csv'
fieldnames = {
    'username',
    'handle',
    'join_date',
    'tweet_count',
    'following_count',
    'followers_count',
    'likes_count',
    'tweets',
    'replies'
}

# with open(csv_file,mode='w',newline='', encoding = 'utf-8')as file:
#     writer = csv.DictWriter(file,fieldnames=fieldnames)
#     writer.writeheader()

with open('list.txt', 'w') as pfile:

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81"
        page = browser.new_page(user_agent=ua)
        #page = context.new_page()

        stealth_sync(page)
        #Wait until some element has loaded
        #page.is_visible('div.tile-body')
        all_profiles=  []
        for query in queries:
            profiles = get_profiles_from_query(page,query)
            for profile in profiles:
                pfile.write(f"{profile}\n")
            all_profiles.extend(profiles)

        #remove duplicates
        all_profiles = list(set(all_profiles))
        print(len(all_profiles))
        # with open('list.txt', 'w') as pfile:
        #     for item in all_profiles:
        #         pfile.write(f"{item}\n")

        # for i in range(len(all_profiles)):
        #     print(i)
        #     profile = all_profiles[i]
        #     p_url = profile+"/with_replies"
        #     try:
        #         page.goto(p_url)
        #     except:
        #         continue
        #     temp = extract_profile_details(page,p_url)
        #     if 'username' not in temp:
        #         continue
        #     writer.writerow(temp)
        


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
