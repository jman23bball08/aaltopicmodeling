def load_emotion_lexicon(file_path):
    emotion_lexicon = {}
    with open(file_path, "r") as file:
        for line in file:
            word, emotion, value = line.strip().split("\t")
            if value == "1":
                if word in emotion_lexicon:
                    emotion_lexicon[word].append(emotion)
                else:
                    emotion_lexicon[word] = [emotion]
    return emotion_lexicon

def calculate_vocabulary_overlap(vocab, emotion_lexicon):
    overlapping_words = set(vocab).intersection(emotion_lexicon.keys())
    overlap_percentage = (len(overlapping_words) / len(vocab)) * 100
    return overlap_percentage
