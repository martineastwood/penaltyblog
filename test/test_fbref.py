# import unittest
# import penaltyblog as pb

# COUNTRY = "England"
# COMPETITION = "Premier League"
# SEASON = "2021-2022"


# class TestFbRef(unittest.TestCase):
#     def test_country(self):
#         countries = pb.fbref.list_countries()
#         self.assertGreater(len(countries), 0)

#         country = pb.fbref.get_country(COUNTRY)
#         self.assertEqual(country.country, COUNTRY)
#         self.assertEqual(country.governing_body, "UEFA")

#         competitions = country.list_competitions()
#         self.assertGreater(len(competitions), 0)
#         self.assertTrue(COMPETITION in competitions)

#     def test_competitions(self):
#         country = pb.fbref.get_country(COUNTRY)
#         competition = country.get_competition(COMPETITION)

#         self.assertEqual(competition.competition_name, COMPETITION)
#         self.assertEqual(competition.country.country, COUNTRY)

#         seasons = competition.list_seasons()
#         self.assertGreater(len(seasons), 0)
#         self.assertTrue(SEASON in seasons)

#     def test_seasons(self):
#         country = pb.fbref.get_country(COUNTRY)
#         competition = country.get_competition(COMPETITION)
#         season = competition.get_season(SEASON)

#         self.assertEqual(season.season_name, SEASON)
#         self.assertEqual(season.competition.competition_name, COMPETITION)
#         self.assertEqual(season.competition.country.country, COUNTRY)

#         tbl = season.get_league_table()
#         self.assertEqual(tbl["Squad"].iloc[0], "Manchester City")
#         self.assertEqual(tbl["Squad"].iloc[20], "Norwich City")


# if __name__ == "__main__":
#     unittest.main()
