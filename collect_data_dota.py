import opendota
import json

# Initialize the OpenDota API client
# The client will automatically cache responses to prevent re-downloading
client = opendota.OpenDota()

# A sample match ID to fetch. You can find these on the OpenDota website.
# This example uses a real, parsed match.
MATCH_ID = '7861033621'

try:
    # Fetch the detailed data for the specified match
    # This corresponds to the /matches/{match_id} endpoint
    print(f"Fetching data for match ID: {MATCH_ID}...")
    match_data = client.get_match(MATCH_ID)
    print("Successfully fetched match data.")

    # Save the full JSON response to a file for inspection
    output_filename = f'match_{MATCH_ID}_details.json'
    with open(output_filename, 'w') as f:
        json.dump(match_data, f, indent=4)
    print(f"Full match data saved to {output_filename}")

    # --- Example of accessing sequential data ---

    # Print the radiant team's gold advantage at each minute of the game
    if 'radiant_gold_adv' in match_data and match_data['radiant_gold_adv']:
        print("\nRadiant Gold Advantage per Minute:")
        # The list index corresponds to the minute of the game
        for minute, gold_adv in enumerate(match_data['radiant_gold_adv']):
            print(f"Minute {minute}: {gold_adv}")
    else:
        print("\nRadiant Gold Advantage data not available for this match.")

    # Print objectives (e.g., tower kills, Roshan kills) with timestamps
    if 'objectives' in match_data and match_data['objectives']:
        print("\nGame Objectives:")
        for objective in match_data['objectives']:
            event_time_minutes = objective['time'] // 60
            event_time_seconds = objective['time'] % 60
            player_slot = objective.get('player_slot', 'N/A')
            objective_type = objective['type']
            print(f"Time: {event_time_minutes:02d}:{event_time_seconds:02d} - Player Slot: {player_slot} - Event: {objective_type}")
    else:
        print("\nObjectives data not available for this match.")

except Exception as e:
    print(f"An error occurred: {e}")