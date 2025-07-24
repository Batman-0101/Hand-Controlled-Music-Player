# Gesture-Based Enhancements
# Volume Fine-Tuning:
#
# Instead of a simple distance threshold, allow users to control volume dynamically by moving their fingers in a circular motion.
# Clockwise: Increase volume.
# Counterclockwise: Decrease volume.
#
# Playback Controls:
# Pause/Play Gesture: A pinch motion (thumb and index finger touching) could toggle pause/play.
# Skip to Specific Points: Swiping left/right with three fingers could allow fast-forwarding or rewinding.
#
# Gesture-Based Equalizer (EQ) Mode:
# Different hand positions could adjust bass, treble, or balance.
# Example: Moving a hand vertically changes bass levels.
#
# Visual Enhancements
# Hand Landmark Heatmap:
# Overlay a heatmap on the screen based on the most frequently detected hand positions.
#
# Customizable UI Elements:
# Allow the user to change colors for different gestures.
#
# Audio Features
# Song Recommendation Based on Gesture History:
# If a user frequently skips a song quickly, reduce its play frequency.
# If a song is played fully without skipping, increase its priority in the playlist.
#
# Real-Time Lyrics Display:
# Use an API to fetch lyrics and display them in sync with the song.
#
# Multiplayer/Interaction Features
# Dual-Hand Control Mode:
# If two users are detected, each can control different aspects.
#
# Example: One controls volume while the other changes the song.
#
# Handwriting Recognition for Song Search:
# Users could write the first letter of a song in the air, and the system searches for matching songs.

import os
import cv2
import time
import collections
import pygame
import HandTrackingModule as htm
import math
from mutagen.mp3 import MP3
import numpy as np
from RealtimeSTT import AudioToTextRecorder


####################################################################################################

def calculate_fps(pTime, cTime):
	cTime = time.time()
	fps = round(1 / (cTime - pTime), 3) if cTime != pTime else 0
	fps_values.append(fps)
	avg_fps = sum(fps_values) / len(fps_values)
	pTime = cTime
	return avg_fps, pTime

##################################################

def get_angle_from_tangent(opposite, adjacent):
	if adjacent == 0:
		if opposite > 0:
			return 90.0
		elif opposite < 0:
			return -90.0
		else:
			return 0
	
	angle_radians = math.atan(opposite / adjacent)
	angle_degrees = math.degrees(angle_radians)
	return angle_degrees

##################################################

def change_song(direction=1):
	"""Change song and return its name."""
	global current_song_index, paused
	
	pygame.mixer.music.stop()
	current_song_index = (current_song_index + direction) % len(songs)
	pygame.mixer.music.load(songs[current_song_index])
	pygame.mixer.music.play()
	paused = False
	
	# Extract clean song name
	return songs[current_song_index].split('/')[-1].replace('.mp3', '')



##################################################

def format_song_title(words_list):
	if not words_list:
		return ""
	
	title = words_list[0]  # First element is the song title
	artist_section = words_list[1:]  # The rest is the artist and features
	
	if 'Ft' in artist_section:
		ft_index = artist_section.index('Ft')
		main_artist = " ".join(artist_section[:ft_index])
		features = " ".join(artist_section[ft_index + 1:])
		return f"{title} by {main_artist} Ft. {features}"
	else:
		main_artist = " ".join(artist_section)
		return f"{title} by {main_artist}"
	
##################################################

def sec_to_min(seconds):
	"""Convert seconds to minutes and seconds."""
	minutes = int(seconds // 60)
	seconds = int(seconds % 60)
	return f"{minutes}:{seconds:02}"


def process_text(text):
    print(text)

####################################################################################################

pygame.mixer.init(frequency=44100)
folderPath = '../songs'
songs = [os.path.join(folderPath, song).replace("\\", "/")
         for song in os.listdir(folderPath) if song.lower().endswith('.mp3')]

current_song_index = 0
paused = False

current_song = change_song()

##################################################

# Video Capture Setup
wCam, hCam = 1080, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

##################################################

# color = (255,255,255)
color = (53,0,0)
seccolor = (255,255,255)
pTime = 0
cTime = 0
dist = 0
Dist2hR = 0
Dist2hL = 0
midXdist = 0
midYdist = 0
angle = 0
angleR = 0
angleP = 0
fps_values = collections.deque(maxlen=30)
detector = htm.handDetector(minDetConf=0.8, minTrackConf=0.8)
changeSongF = False
changeSongB = False
hand = ''
ColorAngle1hP = color
ColorAngle1hN = color
changeSongN1H = False
changeSongP1H = False


####################################################################################################

def main():
	global pTime, cTime, dist, midXdist, midYdist, angle, angleR, angleP, fps_values, current_song, changeSongF, changeSongB, hand, ColorAngle1hP, ColorAngle1hN, changeSongN1H, changeSongP1H, paused
	# print("Wait until it says 'speak now'")
	# recorder = AudioToTextRecorder(wake_words="jarvis")
	
	while True:
		# recorder.text(process_text)
		success, img = cap.read()
		img = detector.findHands(img, draw=False)  # Draw hands on the image
		fingers = detector.numFingers(img, detector.findPos(img, draw=False), draw=False)  # Get the number of fingers
		numHands = detector.numHands(numHands=False)
		lmList = detector.findPos(img, draw=False)
		
		################################################################################################
		

		
		if len(lmList) != 0:
			if numHands == 1:  # If only one hand is detected
				detector.drawLines(img, [[4, 8]], color)  # For drawing the connection between thumb and index fingers
				dist, midXdist, midYdist = detector.findDistance(lmList, lmList, img, 4,
				                                        8)  # Distance between thumb and index fingers
			# print('dist', dist, midXdist, midYdist)
			elif numHands == 2:  # If two hands are detected
				lmList1, lmList2 = lmList[:21], lmList[21:]  # Split the landmarks into two hands
				dist, midXdist, midYdist = detector.findDistance(lmList1, lmList2, img, 5, 5) # Distance between the index fingers of both hands
			
			############################################################################################
			# Handle volume control (1 hand)
			if dist is not None and numHands == 1:
				if sum(fingers) == 0 and paused == False:
					pygame.mixer.music.pause()
					paused = True
				elif sum(fingers) == 5 and paused == True:
					pygame.mixer.music.unpause()
					paused = False

				dist = float(dist * 100)  # Scale the distance for volume control
				# print('dist', dist)
				if dist < 10:
					volume = 0  # Mute the music if distance is less than 5
					pygame.mixer.music.set_volume(volume)
				elif dist > 75:
					volume = 1
					pygame.mixer.music.set_volume(volume)
				else:
					# Normalize the volume for distances between 5 and 75
					volume = (dist - 5) / (75 - 5)  # Normalize to the range 0-1
					pygame.mixer.music.set_volume(volume)
				
				if lmList[17][1] > lmList[3][1]:
					hand = 'LEFT'
				else:
					hand = 'RIGHT'
				
				angle = get_angle_from_tangent(lmList[12][2] - lmList[0][2], lmList[12][1] - lmList[0][1])
				angleVolume = get_angle_from_tangent(lmList[8][2] - lmList[4][2], lmList[8][1] - lmList[4][1])
				angleP = abs(angle)
				angleN = angle
				if angleN > 0:
					angleP = 180 - angleP
				if angleN < 0:
					angleN = 180 + angleN
				if hand == 'LEFT':
					if 0 < angleP < 45:
						if changeSongP1H:
							ColorAngle1hP = (0, 255, 0)
							changeSongP1H = False
							current_song = change_song(-1)
					else:
						changeSongP1H = True  # Reset only when angle exits range
						ColorAngle1hP = color
					
					if 0 < angleN < 45:
						if changeSongN1H:
							ColorAngle1hN = (0, 255, 0)
							changeSongN1H = False
							current_song = change_song(1)
					else:
						changeSongN1H = True  # Reset only when angle exits range
						ColorAngle1hN = color
				elif hand == 'RIGHT':
					if 0 < angleP < 45:
						if changeSongP1H:
							ColorAngle1hP = (0, 255, 0)
							changeSongP1H = False
							current_song = change_song(1)
					else:
						changeSongP1H = True  # Reset only when angle exits range
						ColorAngle1hP = color
					
					if 0 < angleN < 45:
						if changeSongN1H:
							ColorAngle1hN = (0, 255, 0)
							changeSongN1H = False
							current_song = change_song(-1)
					else:
						changeSongN1H = True  # Reset only when angle exits range
						ColorAngle1hN = color
				
				if hand == 'LEFT':
					left_z1h = lmList[19][3]  # Left pinky tip depth
					# Left hand scaling parameters (dramatic effect)
					base_sizeL1h = 0.5  # Minimum base size
					max_sizeL1h = 3.0  # Very large maximum size
					min_sizeL1h = 0.3  # Smallest allowed size
					z_scaleL1h = 0.05  # Aggressive scaling factor
					font_sizeL1h = max(min_sizeL1h, min(max_sizeL1h, base_sizeL1h + (1 / (left_z1h + 0.2) * z_scaleL1h)))
					l_pos = (lmList[17][1] + int(40 * font_sizeL1h), lmList[17][2] - int(20 * font_sizeL1h))
					cv2.putText(img, "L", l_pos, cv2.FONT_HERSHEY_SIMPLEX,
					            font_sizeL1h, color,  # Blue color
					            max(1, int(font_sizeL1h * 2)),  # Thickness scales with size
					            lineType=cv2.LINE_AA)
					yMin = min(lmList, key=lambda x: x[2] if len(x) > 2 else float('-inf'))
					xMin = min(lmList, key=lambda x: x[1] if len(x) > 1 else float('-inf'))
					xMax = max(lmList, key=lambda x: x[1] if len(x) > 1 else float('inf'))
					yMin = yMin[2] + 20
					xMin = xMin[1]
					xMax = xMax[1]
					cv2.ellipse(img, (xMax, yMin), (50, 50), 270, 0, 90, color, 1, cv2.LINE_AA)
					cv2.ellipse(img, (xMin, yMin), (50, 50), 180, 0, 90, color, 1, cv2.LINE_AA)
					cv2.putText(img, 'ROTATE - PREV SONG', (xMax + 50, yMin - 50),
					            cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, f'{int(angleP)}', (xMax + 60, yMin - 20),
					            cv2.FONT_HERSHEY_PLAIN, 1, ColorAngle1hP, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, 'ROTATE - NEXT SONG', (xMin - 200, yMin - 50),
					            cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, f'{int(angleN)}', (xMin - 80, yMin - 20),
					            cv2.FONT_HERSHEY_PLAIN, 1, ColorAngle1hN, 1, lineType=cv2.LINE_AA)
				elif hand == 'RIGHT':
					right_z1h = lmList[19][3]
					# Right hand scaling parameters (subtle effect)
					base_sizeR1h = 0.5  # Medium base size
					max_sizeR1h = 3.0  # Moderate maximum size
					min_sizeR1h = 0.3  # Minimum size
					z_scale1h = 0.05  # Gentle scaling factor
					font_sizeR1h = max(min_sizeR1h, min(max_sizeR1h, base_sizeR1h + (1 / (right_z1h + 0.2) * z_scale1h)))
					yMin = min(lmList, key=lambda x: x[2] if len(x) > 2 else float('-inf'))
					xMin = min(lmList, key=lambda x: x[1] if len(x) > 1 else float('-inf'))
					xMax = max(lmList, key=lambda x: x[1] if len(x) > 1 else float('inf'))
					yMin = yMin[2] + 20
					xMin = xMin[1]
					xMax = xMax[1]
					r_pos = (lmList[17][1] - int(40 * font_sizeR1h), lmList[17][2] - int(20 * font_sizeR1h))
					cv2.putText(img, "R", r_pos, cv2.FONT_HERSHEY_SIMPLEX,
					            font_sizeR1h, color,  # Green color
					            max(1, int(font_sizeR1h * 1.5)), lineType=cv2.LINE_AA)
					cv2.ellipse(img, (xMax, yMin), (50, 50), 270, 0, 90, color, 1, cv2.LINE_AA)
					cv2.ellipse(img, (xMin, yMin), (50, 50), 180, 0, 90, color, 1, cv2.LINE_AA)
					cv2.putText(img, 'ROTATE - NEXT SONG', (xMax + 50, yMin - 50),
					            cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, f'{int(angleP)}', (xMax + 60, yMin - 20),
					            cv2.FONT_HERSHEY_PLAIN, 1, ColorAngle1hP, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, 'ROTATE - PREV SONG', (xMin - 200, yMin - 50),
					            cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, f'{int(angleN)}', (xMin - 80, yMin - 20),
					            cv2.FONT_HERSHEY_PLAIN, 1, ColorAngle1hN, 1, lineType=cv2.LINE_AA)
				
				if abs(angleVolume) > 45:
					cv2.putText(img, f"{volume * 100:.0f}", (midXdist + 20, midYdist + 10), cv2.FONT_HERSHEY_PLAIN, 1,
					            color, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, f"VOLUME", (midXdist + 20, midYdist - 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1,
					            lineType=cv2.LINE_AA)
				elif abs(angleVolume) < 45:
					cv2.putText(img, f"VOLUME", (midXdist - 8, midYdist - 50), cv2.FONT_HERSHEY_PLAIN, 1, color, 1,
					            lineType=cv2.LINE_AA)
					cv2.putText(img, f"{volume * 100:.0f}", (midXdist - 10, midYdist - 20), cv2.FONT_HERSHEY_PLAIN, 1,
					            color, 1, lineType=cv2.LINE_AA)
			
			############################################################################################
			
			elif dist is not None and numHands == 2:
				dist = float(dist * 100)  # Normalize the distance for volume control
				# For two hands, control volume with the distance between the index fingers
				if dist < 10:
					volume = 0
					pygame.mixer.music.set_volume(volume)
				elif dist > 400:
					volume = 1
					pygame.mixer.music.set_volume(volume)
				else:
					volume = (dist - 10) / (400 - 10)
					pygame.mixer.music.set_volume(volume)
				
	
				if lmList[17][1] < lmList[17 + 21][1]:
					left_hand = lmList2
					right_hand = lmList1
					left_label = "LEFT (Prev Song)"
					right_label = "RIGHT (Next Song)"
				else:
					left_hand = lmList1
					right_hand = lmList2
					left_label = "LEFT (Prev Song)"
					right_label = "RIGHT (Next Song)"
	
				# Calculate the angle between the index and thumb fingers of both hands
				angleR = abs(get_angle_from_tangent(right_hand[12][2] - right_hand[0][2], right_hand[12][1] - right_hand[0][1]))
				angleL = abs(get_angle_from_tangent(left_hand[12][2] - left_hand[0][2], left_hand[12][1] - left_hand[0][1]))
				angleVolume2h = get_angle_from_tangent(right_hand[8][2] - left_hand[8][2], right_hand[8][1] - left_hand[4][1])
	
				if angleR < 45 and changeSongF:
					changeSongF = False
					current_song = change_song(1)
				if angleL < 45 and changeSongB:
					changeSongB = False
					current_song = change_song(-1)
				if abs(angleVolume2h) > 45:
					cv2.putText(img, f"{volume * 100:.0f}", (midXdist + 20, midYdist + 10), cv2.FONT_HERSHEY_PLAIN, 1,
					            color, 1, lineType=cv2.LINE_AA)
					cv2.putText(img, f"VOLUME", (midXdist + 20, midYdist - 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1,
					            lineType=cv2.LINE_AA)
				elif abs(angleVolume2h) < 45:
					cv2.putText(img, f"VOLUME", (midXdist - 8, midYdist - 50), cv2.FONT_HERSHEY_PLAIN, 1, color, 1,
					            lineType=cv2.LINE_AA)
					cv2.putText(img, f"{volume * 100:.0f}", (midXdist - 10, midYdist - 20), cv2.FONT_HERSHEY_PLAIN, 1,
					            color, 1, lineType=cv2.LINE_AA)
				
				# Get Z-coordinates (depth) for both hands
				left_z = left_hand[19][3]  # Left pinky tip depth
				right_z = right_hand[19][3]  # Right pinky tip depth
				
				# Left hand scaling parameters
				base_sizeL = 0.5
				max_sizeL = 3.0
				min_sizeL = 0.3
				z_scaleL = 0.05
				
				# Right hand scaling parameters
				base_sizeR = 0.5
				max_sizeR = 3.0
				min_sizeR = 0.3
				z_scaleR = 0.05
				
				# Calculate font sizes with non-linear scaling
				font_sizeL = max(min_sizeL, min(max_sizeL, base_sizeL + (1 / (left_z + 0.2) * z_scaleL)))
				font_sizeR = max(min_sizeR, min(max_sizeR, base_sizeR + (1 / (right_z + 0.2) * z_scaleR)))
	
				# Enhanced left hand label (dramatic)
				l_pos = (left_hand[17][1] + int(40 * font_sizeL), left_hand[17][2] - int(20 * font_sizeL))
				cv2.putText(img, "L", l_pos, cv2.FONT_HERSHEY_SIMPLEX,
				         font_sizeL, color,  # Blue color
				max(1, int(font_sizeL * 2)),  # Thickness scales with size
				lineType = cv2.LINE_AA)
				
				# Enhanced right hand label (subtle)
				r_pos = (right_hand[17][1] - int(40 * font_sizeR), right_hand[17][2] - int(20 * font_sizeR))
				cv2.putText(img, "R", r_pos, cv2.FONT_HERSHEY_SIMPLEX,
				         font_sizeR, color,  # Green color
				max(1, int(font_sizeR * 1.5)), lineType = cv2.LINE_AA)  # Thickness scales with size
				
				Dist2hR, midXR, midYR = detector.findDistance(right_hand, right_hand, img, 4, 8)
				Dist2hL, midXR, midYR = detector.findDistance(left_hand, left_hand, img, 4, 8)
				
				ColorRAngle = color if angleR > 45 else (0, 255, 0)
				ColorLAngle = color if angleL > 45 else (0, 255, 0)
				yMinL = min(left_hand, key=lambda x: x[2] if len(x) > 2 else float('-inf'))[2] + 20
				yMinR = min(right_hand, key=lambda x: x[2] if len(x) > 2 else float('-inf'))[2] + 20
				xMinR = min(right_hand, key=lambda x: x[1] if len(x) > 1 else float('-inf'))[1]
				xMaxL = max(left_hand, key=lambda x: x[1] if len(x) > 1 else float('inf'))[1]
				
				cv2.ellipse(img, (xMaxL, yMinL), (50,50), 270, 0, 90, color, 1, cv2.LINE_AA)
				cv2.putText(img, 'ROTATE - PREV SONG', (xMaxL + 50, yMinL - 50), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
				cv2.putText(img, f'{int(angleL)}', (xMaxL + 60, yMinL - 20 ), cv2.FONT_HERSHEY_PLAIN, 1, ColorLAngle, 1, lineType=cv2.LINE_AA)
	
				cv2.ellipse(img, (xMinR, yMinR), (50,50), 180, 0, 90, color, 1, cv2.LINE_AA)
				cv2.putText(img, 'ROTATE - NEXT SONG', (xMinR - 200, yMinR - 50), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
				cv2.putText(img, f'{int(angleR)}', (xMinR - 80, yMinR - 20 ), cv2.FONT_HERSHEY_PLAIN, 1, ColorRAngle, 1, lineType=cv2.LINE_AA)
	
				# Check angles for song change
				if angleR > 45:
					changeSongF = True
				if angleL > 45:
					changeSongB = True
		
		avg_fps, pTime = calculate_fps(pTime, cTime)
		
		songName = current_song.split('_')
		songName = format_song_title(songName)
		
		cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
		cv2.putText(img, f"{songName}",((img.shape[1] - cv2.getTextSize(f"{songName}", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0][0]) // 2, 500),cv2.FONT_HERSHEY_PLAIN, 2, color, 1, lineType=cv2.LINE_AA)
		
		songPlaying = MP3('../songs/' + current_song + '.mp3')
		songLength = songPlaying.info.length
		songPlayTime = pygame.mixer.music.get_pos() / 1000
	
		songBarType = 1
	
		if songBarType == 1:
			songBarY = 520
			songPlayLength = np.interp(songPlayTime, [0, songLength], [100, img.shape[1]-100])
			cv2.line(img, (100, songBarY), (img.shape[1]-100, songBarY), color, 2, lineType=cv2.LINE_AA)
			cv2.line(img, (100, songBarY), (int(songPlayLength), songBarY), seccolor, 2, lineType=cv2.LINE_AA)
			cv2.putText(img, f"{sec_to_min(songLength)}", (img.shape[1] - 90, songBarY+4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
			cv2.putText(img, f"{sec_to_min(songPlayTime)}", (55, songBarY+4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
		elif songBarType == 2:
			songBarY = 538
			songPlayLength = np.interp(songPlayTime, [0, songLength], [0, img.shape[1]])
			cv2.line(img, (0, songBarY), (img.shape[1], songBarY), color, 5, lineType=cv2.LINE_AA)
			cv2.line(img, (0, songBarY), (int(songPlayLength), songBarY), seccolor, 5, lineType=cv2.LINE_AA)
			cv2.putText(img, f"{sec_to_min(songPlayTime)} / {sec_to_min(songLength)}", (img.shape[1]//2 - 50, songBarY-15), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, lineType=cv2.LINE_AA)
			
		if sec_to_min(songPlayTime) == sec_to_min(songLength):
			current_song = change_song(1)
			songName = current_song.split('_')
			songName = format_song_title(songName)
	
		cv2.imshow("Image", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break  # Quit when 'q' is pressed


if __name__ == "__main__":
	main()
