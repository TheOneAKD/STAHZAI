from astroquery.simbad import Simbad

# Set up the Simbad query
query = "star name"  # Replace "star name" with the actual name of the star
criteria = Simbad.criteria.MAIN_ID  # Use criteria for desired columns

# Execute the query
result = Simbad.query_criteria(query, criteria=criteria)

# Print the results
print(result)