#BP_NPCCharacter_C_UAID_00D8617686C4E4FD02_2128565794


import unrealcv

# Connect to UnrealCV
client = unrealcv.Client(('127.0.0.1', 9000))
client.connect()

if not client.isconnected():
    print("Failed to connect to UnrealCV")
    exit()

print("Connected to UnrealCV!")

# Vector3 target position
target = (1455.499, -208.124, 190.67)

# Send Vector3 to BP_NPCCharacter
command = f"vbp BP_WAM_Gaurd_C_1 MoveTo (X={target[0]},Y={target[1]},Z={target[2]})"

response = client.request(command)

print("Vector3 sent to BP_NPCCharacter:", target)
print("UnrealCV response:", response)

client.disconnect()

