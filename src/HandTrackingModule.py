import cv2
import mediapipe as mp
import collections
import time


class handDetector():
	def __init__(self, mode=False, maxHands=2, modComp=1, minDetConf=0.5, minTrackConf=0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.modelComplexity = modComp
		self.minDetectionConfidence = minDetConf
		self.minTrackingConfidence = minTrackConf
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.minDetectionConfidence,
		                                self.minTrackingConfidence)
		self.mpDraw = mp.solutions.drawing_utils
		self.lmList = []  # Initialize lmList
	
	def findHands(self, img, draw=True):
		imqRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imqRGB)
		
		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
		
		return img
	
	def numHands(self, numHands=True):
		if numHands and self.results.multi_hand_landmarks:
			print('# of hands detected: ', len(self.results.multi_hand_landmarks))
		if self.results.multi_hand_landmarks:
			return len(self.results.multi_hand_landmarks)
		else:
			return 0
	
	def findPos(self, img, draw=True):
		lmList = []
		
		if self.results.multi_hand_landmarks:
			for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
				for id, lm in enumerate(handLms.landmark):
					h, w, c = img.shape
					cx, cy = int(lm.x * w), int(lm.y * h)
					cz = lm.z
					lmList.append([id, cx, cy, cz])  # Add all landmarks for each hand
				
				# Optionally, draw circles on the landmarks
				if draw:
					for id, lm in enumerate(handLms.landmark):
						cx, cy = int(lm.x * w), int(lm.y * h)
						cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)
		
		self.lmList = lmList
		return lmList
	
	def drawLines(self, img, handNos, color=(255,255,255)):
		if self.results.multi_hand_landmarks:
			h, w, _ = img.shape  # Get image dimensions
			
			# Loop through detected hands
			for hand_idx, handLms in enumerate(self.results.multi_hand_landmarks):
				# Ensure lmList is populated before accessing it
				if not self.lmList:  # Only update it if empty
					self.findPos(img)  # Update lmList with the current hand's landmarks
				
				for handNo in handNos:  # Loop through the pairs of landmarks to connect
					lm1, lm2 = handNo[0], handNo[1]  # Get the two landmarks
					# Ensure indices are within the range of landmarks (21 per hand)
					if lm1 >= len(self.lmList) or lm2 >= len(self.lmList):
						continue  # Skip invalid indices
					
					# Convert normalized coordinates to pixel coordinates
					x1, y1 = int(self.lmList[lm1][1]), int(self.lmList[lm1][2])
					x2, y2 = int(self.lmList[lm2][1]), int(self.lmList[lm2][2])
					
					# Check if coordinates are within image bounds
					if x1 >= w or y1 >= h or x2 >= w or y2 >= h:
						print(f"Warning: Coordinates out of bounds for line: ({x1}, {y1}) -> ({x2}, {y2})")
						continue  # Skip drawing if coordinates are out of bounds
					
					# Draw the line
					cv2.line(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
					
					# Draw circles at the landmarks
					cv2.circle(img, (x1, y1), 10, color, 2, lineType=cv2.LINE_AA)
					cv2.circle(img, (x2, y2), 10, color, 2, lineType=cv2.LINE_AA)
		
		return img
	
	def findDistance(self, lmList1, lmList2, img, p1, p2, normalize=True):
		if len(self.lmList) >= 9:  # Ensure at least 9 landmarks exist
			lm1, lm2 = lmList1[p1], lmList2[p2]  # Specific landmarks (e.g., index and thumb tip)
			x1, y1, z1 = lm1[1], lm1[2], lm1[3]
			x2, y2, z2 = lm2[1], lm2[2], lm2[3]
			
			# Calculate 3D Euclidean Distance between the two landmarks (inside hand)
			dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
			
			if normalize:
				# Normalize by hand size (distance from wrist (0) to middle fingertip (12))
				if len(self.lmList) > 12:
					x0, y0, z0 = self.lmList[0][1], self.lmList[0][2], self.lmList[0][3]  # Wrist
					x12, y12, z12 = self.lmList[12][1], self.lmList[12][2], self.lmList[12][3]  # Middle fingertip
					handSize = ((x0 - x12) ** 2 + (y0 - y12) ** 2 + (z0 - z12) ** 2) ** 0.5
					if handSize > 0:  # Avoid division by zero
						dist /= handSize
			
			# Display distance on screen
			midX, midY = (x1 + x2) // 2, (y1 + y2) // 2
			
			return dist, midX, midY
		return None
	
	def numFingers(self, img, lmList, draw=True):
		tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
		if len(lmList) != 0:
			fingers = []
			for id in range(0,5):
				if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
					fingers.append(1)
				else:
					fingers.append(0)
				if lmList[4][1] < lmList[2][1]:
					fingers[0] = 0
				else:
					fingers[0] = 1
			return fingers
	


def main():
	pTime = 0
	cTime = 0
	fps_values = collections.deque(maxlen=30)
	cap = cv2.VideoCapture(0)
	detector = handDetector()
	
	while True:
		success, img = cap.read()
		img = detector.findHands(img)
		
		num_hands = detector.numHands()  # Get the number of hands detected
		print(f'Number of hands detected: {num_hands}')
		
		if num_hands > 0:
			lmList = detector.findPos(img)  # Get landmarks for all detected hands
			if len(lmList) != 0:
				print(f"Total number of landmarks: {len(lmList)}")  # Should be 42 for two hands
				
				# You can access landmarks for individual hands from lmList if needed
				for i in range(0, len(lmList), 21):
					print(f"Landmarks for hand: {lmList[i:i + 21]}")  # Print landmarks for each hand
		
		# Calculate FPS
		cTime = time.time()
		fps = round(1 / (cTime - pTime), 3) if cTime != pTime else 0
		fps_values.append(fps)
		avg_fps = sum(fps_values) / len(fps_values)
		pTime = cTime
		
		cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, lineType=cv2.LINE_AA)
		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == "__main__":
	main()
