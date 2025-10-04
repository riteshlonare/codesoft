import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define genres
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']

# Dummy training data (example plots for each genre)
train_plots = [
    "A hero fights villains in an epic battle.",  # Action
    "Friends make jokes and laugh all day.",     # Comedy
    "A deep story about love and loss.",         # Drama
    "A scary ghost haunts a house.",             # Horror
    "Two people fall in love.",                  # Romance
    "Aliens invade Earth.",                      # Sci-Fi
    "A detective solves a mystery.",             # Thriller
]
train_labels = [0, 1, 2, 3, 4, 5, 6]  # Corresponding to genres indices

# Test data (pasted from user-provided content)
test_data = """1 ::: Edgar's Lunch (1998) ::: L.R. Brane loves his life - his car, his apartment, his job, but especially his girlfriend, Vespa. One day while showering, Vespa runs out of shampoo. L.R. runs across the street to a convenience store to buy some more, a quick trip of no more than a few minutes. When he returns, Vespa is gone and every trace of her existence has been wiped out. L.R.'s life becomes a tortured existence as one strange event after another occurs to confirm in his mind that a conspiracy is working against his finding Vespa.
2 ::: La Haine (1995) ::: 19 May 1993. The day of his friend's release from prison. Vincent has promised to wait for him in front of the station at 11 o'clock. In the meantime, he looks for his pal Hubert, a boxer who is preparing for a fight. But the day takes a different turn when a policeman loses his temper and shoots a young Arab in the head. From then on, the day is marked by tension between young people and the police in the Paris suburbs. The film follows three young men and their descent into violence.
3 ::: Office Killer (1997) ::: When Dorothy Parker was a child, she was traumatized when her mother left her at a summer camp. Years later, she works as a secretary at a failing office. When her boss is fired, she takes over his job. But when she is passed over for a promotion, she snaps and begins killing her co-workers.
4 ::: La Chienne (1931) ::: Cashier Maurice Legrand is married to Adele, a terror. He finds comfort in painting and a prostitute, Lulu. When Lulu's pimp demands more money, Maurice kills him. Lulu blackmails Maurice into leaving his wife and marrying her. But Lulu is not satisfied with Maurice's money and starts an affair with Dédé, a gigolo. Maurice kills Lulu and Dédé, but Adele finds out and blackmails him.
5 ::: Cyclo (1995) ::: In Saigon, a cyclo driver named Poet lives with his sister and her baby. He borrows money from a gangster to buy a new cyclo, but when it's stolen, he has to work for the gangster. The gangster forces Poet to transport drugs and kill people. Poet's life spirals into violence and despair.
6 ::: The Green Mile (1999) ::: Paul Edgecomb, a prison guard on death row in the 1930s, meets John Coffey, a giant black man convicted of raping and murdering two white girls. Paul discovers that John has healing powers. As Paul investigates, he uncovers the truth about John's innocence.
7 ::: Reservoir Dogs (1992) ::: After a botched diamond heist, six criminals are on the run. They meet in a warehouse to divide the loot, but tensions rise as they suspect a rat among them. The film explores their paranoia and violence.
8 ::: The Usual Suspects (1995) ::: A sole survivor of a boat explosion tells the story of a criminal mastermind named Keyser Söze. The police try to piece together the events leading to the explosion.
9 ::: The Godfather (1972) ::: The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.
10 ::: The Shawshank Redemption (1994) ::: Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."""

# Parse test data into list of (id, title, plot)
test_entries = []
for line in test_data.strip().split('\n'):
    parts = line.split(' ::: ')
    if len(parts) == 3:
        id_ = parts[0]
        title = parts[1]
        plot = parts[2]
        test_entries.append((id_, title, plot))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocess training and test data
train_processed = [preprocess(p) for p in train_plots]
test_processed = [preprocess(entry[2]) for entry in test_entries]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_processed)
X_test = vectorizer.transform(test_processed)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train, train_labels)

# Predict genres for test data
predictions = clf.predict(X_test)

# Output results
print("Genre Classification Results:")
for i, (id_, title, _) in enumerate(test_entries):
    genre = genres[predictions[i]]
    print(f"{id_} ::: {title} ::: {genre}")
