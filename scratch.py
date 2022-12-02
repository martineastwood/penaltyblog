import penaltyblog as pb

##

fd = pb.scrapers.FootballData("ENG Premier League", "2014-2015")
df = fd.get_fixtures()
