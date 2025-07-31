import minerl
import numpy as np

# The environment name must match the one you downloaded
MINERL_ENV_NAME = 'MineRLObtainDiamond-v0'
# The data_dir must point to the parent directory of the environment folder
MINERL_DATA_DIR = '/path/to/your/minerl_data'

print(f"Loading data for {MINERL_ENV_NAME} from {MINERL_DATA_DIR}...")

# Create a data pipeline for the specified environment [2]
data = minerl.data.make(MINERL_ENV_NAME, data_dir=MINERL_DATA_DIR)

# Use the BufferedBatchIter for efficient data loading [3]
# This iterates through sequences of state-action-reward tuples
iterator = minerl.data.BufferedBatchIter(data)

# --- Iterate through the data and access detailed state information ---

# We will iterate through 5 sequences of 10 steps each as an example
# A full run would use num_epochs=-1 to loop indefinitely
print("\n--- Sampling 5 sequences of 10 steps each ---")
for i, (current_state, action, reward, next_state, done) in enumerate(iterator.buffered_batch_iter(batch_size=1, seq_len=10, num_epochs=1)):
    if i >= 5:
        break

    print(f"\n--- SEQUENCE {i+1} ---")

    # The 'current_state' is a dictionary of observations at each step in the sequence.
    # We can access specific data points by their keys.

    # Example: Access player's XYZ position from 'location_stats'
    # The shape is (sequence_length, 3) for (x, y, z)
    player_positions = current_state['location_stats']['pos']
    print(f"Player Position at first step: {player_positions}")
    print(f"Player Position at last step: {player_positions[-1]}")


    # Example: Access player's life stats (health, food, etc.)
    # These are dictionaries where values are arrays of shape (sequence_length,)
    player_health = current_state['life_stats']['life']
    player_food = current_state['life_stats']['food']
    print(f"Player Health at first step: {player_health}")
    print(f"Player Food level at last step: {player_food[-1]}")


    # Example: Access the player's inventory
    # This shows the count of 'cobblestone' in the inventory over the sequence
    cobblestone_counts = current_state['inventory']['cobblestone']
    print(f"Cobblestone in inventory at first step: {cobblestone_counts}")


    # Example: Access the action taken by the human player
    # 'action' is a dictionary of all possible actions at each step
    camera_movement = action['camera']
    is_jumping = action['jump']
    print(f"Camera movement at first step: {camera_movement}")
    print(f"Was player jumping at last step? {'Yes' if is_jumping[-1] == 1 else 'No'}")

# Remember to close the data pipeline when you are done
data.close()
print("\nData pipeline closed.")