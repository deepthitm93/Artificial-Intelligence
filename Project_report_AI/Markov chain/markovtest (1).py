import time
import random

class MarkovChain:
	def __init__(self):
		self.textData = []
		self.markov = []

	def parseWords(self, words):
		self.textData = words
		self.prepare()

	def parseTextFile(self, textFilePath):
		with open(textFilePath) as file:
			self.textData = file.read()
		self.prepare()

	def prepare(self):
		self.textData = [i.lower() for i in self.textData.split(" ") if i.isalpha()]
		self.markov = {i:[] for i in self.textData}
		for before, after in zip(self.textData, self.textData[1:]):
			self.markov[before].append(after)
		
	def generate(self):
		new = list(self.markov.keys())
		seed = random.randrange(len(new))
		currentWord = random.choice(new)
		sentence = [currentWord]
		for i in range(0, random.randrange(15, 30)):
			check = self.markov[currentWord]
			if (len(check) > 0):
				nextWord = random.choice(check)
				sentence.append(nextWord)
				currentWord = nextWord
			else:
				currentWord = random.choice(new)
		return " ".join(sentence)

def main():
	testStrings =  "That's where with a cable comes in. You secure it to something that'd require them to have tools to cut the cable, which is " \
		"slightly less likely. Most burglaries are fairly quick, because they make noise and every minute spent inside gets the burglar " \
		"a minute closer to getting caught. The kind of burglar who doesn't get caught tries to spend less than 3 minutes on-premises. " \
		"Bringing along a toolbox increases noise, decreases agility, and makes it harder to carry fenceable items away. So they don't " \
		"tend to have a nice pair of bolt cutters unless they're stupid or know in advance something valuable requires them. It's not " \
		"foolproof, but it's a way to increase the odds the burglar won't be able to steal some things you'd really rather them not " \
		"steal. Sort of like putting locks on the door. Some \"burglars\" try every knob they see and are more than happy to enter an " \
		"unlocked car/residence. But others are willing to kick the door in or break a window, taking added risk they'll be caught. " \
		"That doesn't make a lock \"useless\", it means the lock protects you from some unknown % of risk. That's what the safe with a " \
		"cable does: it shaves some percentage points of your risk away."
	m = MarkovChain()
	m.parseWords(testStrings)
	startTime = time.monotonic()
	while (time.monotonic() - startTime < 4):
		print(m.generate())
		time.sleep(1)


if __name__ == "__main__":
	main()
