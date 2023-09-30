import penaltyblog as pb

fbref = pb.scrapers.FBRef("NLD Eredivisie", "2023-2024")
df = fbref.get_stats("standard")

print(df["players"]["born"].unique())
