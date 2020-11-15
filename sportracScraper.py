from bs4 import BeautifulSoup
import requests
import re

TEAMS = ["New-York-Yankees", "Los-Angeles-Dodgers", "Chicago-Cubs", "Boston-Red-Sox", "Houston-Astros", "New-York-Mets",
         "Philadelphia-Phillies", "Washington-Nationals", "San-Francisco-Giants", "St.-Louis-Cardinals",
         "San-Diego-Padres", "Colorado-Rockies", "Los-Angeles-Angels", "Arizona-Diamondbacks", "Texas-Rangers",
         "Atlanta-Braves", "Minnesota-Twins", "Cincinnati-Reds", "Toronto-Blue-Jays", "Chicago-White-Sox",
         "Seattle-Mariners", "Detroit-Tigers", "Milwaukee-Brewers", "Cleveland-Indians", "Oakland-Athletics",
         "Kansas-City-Royals", "Miami-Marlins", "Tampa-Bay-Rays", "Pittsburgh-Pirates", "Baltimore-Orioles"]


# The request URL is https://www.spotrac.com/mlb/{Team}/payroll/2020/

def requestPayroll(team: str) -> str:
    return requests.get(f"https://www.spotrac.com/mlb/{team}/payroll/2020/").text


def parseRow(tr):
    name = tr.find('a').text
    attributes = tr.select('td > span.cap')
    age = attributes[0].text
    pos = attributes[1].text
    status = attributes[2].text
    salary = attributes[6].text  # .replace(',', '').replace('$', '').replace(' ', '')
    salary_regex = re.sub(r',|\$|\s|-', '', salary)

    return [name, age, pos, status, salary_regex]


def generateTeamTable(html, team):
    """
    :param html: The HTML page for a teams payroll
    :return: a CSV of Player Name, Age, Pos, Status, Salary, Team
    """
    salaries = []
    soup = BeautifulSoup(html, "html.parser")
    # the Player Rows can be found with the query 'td.player>a' (this is hacky but it filters to only clickable players
    # since sporttrac uses the class player for other styling within tables.
    playerLinks = soup.select('td.player>a')
    # Since we got the link of players, the row with all the information is 2 up in the parent tree
    playerRows = [playerLink.parent.parent for playerLink in playerLinks]
    for playerRow in playerRows:
        entry = parseRow(playerRow)
        entry.append(team)
        if not (entry[3].startswith('$') or entry[3].startswith('-')):
            salaries.append(entry)
    return salaries


def requestAndTable():
    allSalries = []
    for team in TEAMS:
        allSalries.append(generateTeamTable(requestPayroll(team), team))
    file = open("data/salaries.csv","w")
    file.write("Name,Age,Pos,Arb.Status,Salary,Team\n")
    for team in allSalries:
        for player in team:
            entry = ",".join(player) + '\n'
            file.write(entry)
    file.close()

requestAndTable()