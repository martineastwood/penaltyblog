class SanitizedTeamNames(dict):
    @staticmethod
    def __missing__(key):
        return key


sanitized_team_names = SanitizedTeamNames()
sanitized_team_names["Bayern"] = "Bayern Munich"
sanitized_team_names["Birmingham"] = "Birmingham City"
sanitized_team_names["Blackburn"] = "Blackburn Rovers"
sanitized_team_names["Bolton"] = "Bolton Wanderers"
sanitized_team_names["Bournemouth"] = "AFC Bournemouth"
sanitized_team_names["Brighton"] = "Brighton & Hove Albion"
sanitized_team_names["Cardiff"] = "Cardiff City"
sanitized_team_names["Derby"] = "Derby County"
sanitized_team_names["Huddersfield"] = "Huddersfield Town"
sanitized_team_names["Leeds"] = "Leeds United"
sanitized_team_names["Leicester"] = "Leicester City"
sanitized_team_names["Luton"] = "Luton Town"
sanitized_team_names["Man City"] = "Manchester City"
sanitized_team_names["Man United"] = "Manchester United"
sanitized_team_names["Man Utd"] = "Manchester United"
sanitized_team_names["Newcastle"] = "Newcastle United"
sanitized_team_names["Norwich"] = "Norwich City"
sanitized_team_names["Nott'm Forest"] = "Nottingham Forest"
sanitized_team_names["Preston"] = "Preston North End"
sanitized_team_names["QPR"] = "Queens Park Rangers"
sanitized_team_names["Rotherham"] = "Rotherham United"
sanitized_team_names["Sheffield Weds"] = "Sheffield Wednesday"
sanitized_team_names["Spurs"] = "Tottenham Hotspur"
sanitized_team_names["Stockport"] = "Stockport County"
sanitized_team_names["Stoke"] = "Stoke City"
sanitized_team_names["Swansea"] = "Swansea City"
sanitized_team_names["Tottenham"] = "Tottenham Hotspur"
sanitized_team_names["Tranmere"] = "Tranmere Rovers"
sanitized_team_names["Wolves"] = "Wolverhampton Wanderers"
sanitized_team_names["Wycombe"] = "Wycombe Wanderers"


def santize_team_names(df):
    """
    Makes team names consistent, e.g. converts Man United to Manchester United etc
    """
    if "team" in df.columns:
        df["team"] = df["team"].map(sanitized_team_names)

    if "team_home" in df.columns:
        df["team_home"] = df["team_home"].map(sanitized_team_names)

    if "team_away" in df.columns:
        df["team_away"] = df["team_away"].map(sanitized_team_names)
    return df
