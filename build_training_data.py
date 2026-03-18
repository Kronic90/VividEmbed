"""
build_training_data.py — Training Data Generator for VividEmbedder
===================================================================
Generates multi-objective training data for fine-tuning a sentence
transformer into a purpose-built companion memory embedding model.

Training objectives:
  1. Emotional congruence    — same-emotion memories cluster together
  2. Semantic similarity     — standard text retrieval quality preserved
  3. Contradiction awareness — opposing sentiments push apart
  4. Mood-conditioned retrieval — mood prefix shifts query embedding
  5. Vividness magnitude     — vector norm encodes importance

Novel conditioning tokens:
  [EMO:x]  — emotion tag on stored memories
  [IMP:n]  — importance level (1–10) on stored memories
  [MOOD:x] — current mood state on retrieval queries
  [QUERY]  — marks a retrieval query vs a stored memory

Output: HuggingFace Datasets saved to   ./vivid_training_data/
"""

from __future__ import annotations

import random
import json
import itertools
from pathlib import Path
from collections import defaultdict

random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "vivid_training_data"

# ══════════════════════════════════════════════════════════════════
# 1.  EMOTION-TAGGED MEMORY DATABASE
#     Rich, diverse first-person memories an AI companion user
#     might generate, grouped by primary emotion.
# ══════════════════════════════════════════════════════════════════

MEMORIES: dict[str, list[str]] = {
    # ── Positive-Active ────────────────────────────────────────
    "happy": [
        "Had an amazing day at the beach with friends, the sun was perfect",
        "Got the promotion I've been working toward for two years",
        "My daughter took her first steps today and I started crying happy tears",
        "Cooked dinner for everyone and they all said it was the best meal",
        "Found out my favourite band is coming to town next month",
        "The whole family got together for the holidays and it was wonderful",
        "Laughed so hard with my best friend that my stomach hurt",
        "Finished painting the living room and it looks incredible",
        "Woke up feeling genuinely cheerful for the first time in weeks",
        "Our team won the tournament and celebrated together afterwards",
        "Spent the afternoon playing board games and everyone was in such a good mood",
        "Received a heartfelt thank-you note from a coworker I helped",
    ],
    "excited": [
        "Just booked flights to Japan, I've been dreaming about this trip for years",
        "Started a new project at work that uses cutting-edge technology",
        "My partner and I are finally moving in together next month",
        "Got accepted into the graduate program I applied to",
        "The new video game I've been waiting for launches tomorrow",
        "My side business just got its first paying customer",
        "Found out we're having a baby and I can't stop smiling",
        "Signed up for my first marathon, training starts next week",
        "Got tickets to the championship game, best seats in the house",
        "My research paper was accepted for publication",
    ],
    "proud": [
        "Ran my first 5K without stopping and beat my target time",
        "My son graduated with honours and gave the valedictorian speech",
        "Finally paid off all my student loans after eight years of payments",
        "Built a piece of furniture from scratch and it actually looks professional",
        "Mentored a junior developer who just got promoted thanks to our work together",
        "Stood up for myself in a meeting when I would have stayed quiet before",
        "Lost thirty pounds over the past year through discipline and consistency",
        "Published my first short story in a literary magazine",
        "Taught myself to play a complete piano piece by ear",
        "Coached the kids' soccer team to their first winning season",
    ],
    "joyful": [
        "The surprise birthday party my friends threw completely overwhelmed me with happiness",
        "Watching the sunrise from the mountain summit after a long night hike was pure bliss",
        "My dog greeted me at the door after I was away for two weeks and wouldn't stop wagging",
        "Dancing in the rain with my partner felt like a scene from a movie",
        "Holding my newborn nephew for the first time filled me with indescribable warmth",
        "The cherry blossoms in the park were breathtaking, I sat there for an hour just watching",
        "Finally reconnected with my childhood best friend after fifteen years apart",
        "The crowd sang along to my favourite song at the concert and I felt part of something",
    ],
    "loving": [
        "My partner left a sweet handwritten note in my lunch bag this morning",
        "Cuddled on the couch watching old movies and felt completely at peace",
        "My child made me a handmade birthday card with glitter everywhere",
        "Told my parents I love them and my dad actually teared up a little",
        "Our anniversary dinner was simple but being together made it perfect",
        "My cat curled up on my chest purring and I felt so appreciated",
        "Spent the evening on a long phone call with my sister just catching up",
        "My best friend drove three hours just to be there for my surgery",
    ],
    "grateful": [
        "My neighbour shovelled my driveway without being asked after the snowstorm",
        "The doctor said the biopsy came back clean and I've never felt more thankful",
        "A stranger helped me change my flat tire in the pouring rain",
        "My mentor spent an hour giving me career advice that genuinely changed my path",
        "Received a scholarship that means I won't have to worry about tuition",
        "My employer gave me flexible hours so I could care for my sick mother",
        "A friend anonymously paid for my groceries when I was struggling financially",
        "The firefighters saved our cat during the house fire and I'll never forget it",
    ],
    # ── Positive-Calm ──────────────────────────────────────────
    "calm": [
        "Sat by the lake watching the water ripple in the late afternoon light",
        "Meditated for twenty minutes and felt my mind settle into stillness",
        "Read a book in the garden while sipping chamomile tea",
        "Listened to rain on the roof while wrapped in a warm blanket",
        "Walked through the forest trail without any destination in mind",
        "Practiced yoga at sunrise and felt completely centred",
        "Watched the clouds drift across the sky from the hammock",
        "Had a quiet evening alone with no screens and just my journal",
    ],
    "content": [
        "Life feels balanced right now, work is manageable and home is good",
        "Finished all my errands early and had the whole afternoon free",
        "Made a simple pasta dinner and enjoyed eating it slowly",
        "My garden is finally thriving after months of patient watering",
        "Spent the morning tidying the house and felt productive and settled",
        "Had a regular Tuesday but realised that regular is actually nice",
        "Took the dog for our usual walk and appreciated the routine",
        "Sat on the porch with a cup of coffee and felt at ease",
    ],
    "hopeful": [
        "Started therapy last week and already feel like things could get better",
        "The job market is picking up and I have three interviews lined up",
        "After a rough year, I can finally see some light ahead",
        "My recovery is going well and the doctor says I'm ahead of schedule",
        "The new treatment option sounds promising, I'm cautiously optimistic",
        "Met someone new and for the first time in a while I feel excited about dating",
        "Planted seeds today and imagined what the garden will look like in spring",
        "My daughter started eating better and her energy is improving",
    ],
    "nostalgic": [
        "Found old photos from college and remembered how carefree things were",
        "Drove past my childhood home and all the memories came flooding back",
        "Heard a song on the radio that took me straight back to high school summers",
        "Made my grandmother's cookie recipe and the kitchen smelled just like hers",
        "Went through my old journals from ten years ago and barely recognise that person",
        "Visited the playground where we used to hang out as kids, it felt bittersweet",
        "Watched a movie from the nineties that reminded me of family movie nights",
        "Found a letter from my late grandfather in an old shoebox",
    ],
    # ── Negative-Low Arousal ───────────────────────────────────
    "sad": [
        "Got the news that my uncle passed away, I wish I'd visited more often",
        "My best friend said she's moving across the country and I already miss her",
        "Came home to an empty apartment and the silence felt crushing",
        "Watched an old video of my dad before he got sick and I couldn't stop crying",
        "The anniversary of my dog's passing hit harder this year than I expected",
        "Tried calling my mum but she didn't pick up and I felt hollow inside",
        "Spent the whole weekend alone without talking to anyone",
        "Read through old text messages from a friendship that fell apart",
        "Cleaned out my late grandmother's closet and found her favourite scarf",
        "The miscarriage has been the hardest thing I've ever gone through",
    ],
    "lonely": [
        "Everyone at work went to lunch together and nobody invited me",
        "Scrolled through social media seeing everyone's plans while I stayed home",
        "Moved to a new city and haven't made a single friend in three months",
        "My roommate has their partner over constantly and I feel invisible",
        "Sat at a coffee shop alone and watched other people laughing in groups",
        "Called three friends to hang out and all of them were busy",
        "Went to a party hoping to connect but left feeling more alone than before",
        "The holidays are coming and I have no family nearby to celebrate with",
    ],
    "melancholy": [
        "Autumn makes me feel a soft kind of sadness I can't quite explain",
        "Looked out the window at the grey sky and let the heaviness wash over me",
        "Played a slow piano piece and felt the weight of old memories",
        "The fading light in the evening always fills me with a quiet ache",
        "Walked the same path we used to walk together and felt the absence",
        "Read a beautiful poem that made me feel deeply and inexplicably sad",
        "Watched the last leaf fall from the tree outside my window",
        "The empty chair at the dinner table still gets to me sometimes",
    ],
    "disappointed": [
        "My project got cancelled after months of hard work",
        "Didn't get the role I auditioned for even though I nailed the callback",
        "The vacation we'd been planning fell through at the last minute",
        "My friend forgot my birthday completely, not even a text",
        "The restaurant I was looking forward to was mediocre and overpriced",
        "My performance review was average when I expected to be recognised",
        "The movie everyone recommended turned out to be terrible",
        "Opened my exam results and they were lower than I studied for",
    ],
    "guilty": [
        "Snapped at my partner over something small and I saw the hurt in their eyes",
        "Forgot my daughter's school play and she didn't say anything, which was worse",
        "Ate the last piece of cake my roommate was saving, felt terrible about it",
        "Said something behind a friend's back that I would hate for them to hear",
        "Missed my mother's call three times this week because I was busy",
        "Promised to help my friend move but cancelled at the last minute",
        "Let my team down by not finishing my part of the project on time",
        "Didn't visit my grandmother in the hospital as often as I should have",
    ],
    # ── Negative-High Arousal ──────────────────────────────────
    "angry": [
        "My coworker took credit for my idea in front of the entire team",
        "The landlord raised the rent again and won't fix anything",
        "Someone cut me off in traffic and then had the nerve to honk at me",
        "Found out my ex has been spreading lies about me to mutual friends",
        "The company laid off half the team but executives got bonuses",
        "My package was stolen from my porch and the delivery company won't help",
        "Got charged a hidden fee on my phone bill that they refused to reverse",
        "My neighbour's dog won't stop barking at three in the morning",
        "Was told to work overtime again with no extra pay",
        "Someone vandalised my car in the parking lot overnight",
    ],
    "frustrated": [
        "Been debugging the same error for six hours and it makes no sense",
        "The insurance company keeps sending me in circles with no resolution",
        "Tried to assemble the furniture and the instructions are completely wrong",
        "My internet has dropped four times during an important meeting today",
        "Explained the same thing to my coworker for the third time this week",
        "The train was cancelled again, that's the fifth time this month",
        "Can't get my code to compile and the error messages are useless",
        "Applied to twenty jobs and haven't heard back from a single one",
    ],
    "anxious": [
        "Have a major presentation tomorrow and I can't stop rehearsing in my head",
        "My medical test results won't be back for a week and the waiting is agony",
        "Every time my phone rings I get a jolt of dread in my stomach",
        "Lying awake at three in the morning worrying about things I can't control",
        "The thought of the upcoming flight makes my palms sweat",
        "I keep checking my email obsessively waiting for the acceptance letter",
        "Heard a strange noise in the house and my heart started racing",
        "Meeting new people at the party tomorrow fills me with dread",
        "Haven't heard from my friend in days and I'm spiralling with worry",
        "The financial situation is keeping me up at night with constant worry",
    ],
    "stressed": [
        "Three deadlines tomorrow and I haven't started on any of them",
        "Juggling work, family obligations, and studying for my certification",
        "My inbox has four hundred unread emails and every one needs a response",
        "The workload keeps increasing but they won't hire anyone to help",
        "Trying to plan the wedding, renovate the house, and keep my job",
        "Running on four hours of sleep for the third day in a row",
        "My mother needs daily care and I'm the only one available to help",
        "The project scope keeps expanding but the deadline hasn't moved",
    ],
    "furious": [
        "Discovered my trusted colleague has been undermining me to management for months",
        "The drunk driver who hit my car walked away with a slap on the wrist",
        "My landlord entered my apartment without permission while I was away",
        "Found messages on my partner's phone that confirmed my worst fears",
        "The scammer stole eight thousand dollars from my elderly father",
        "Was passed over for promotion in favour of someone with half my experience",
        "The school did nothing when my child reported being bullied repeatedly",
        "They demolished the community garden to build another parking lot",
    ],
    # ── Neutral / Complex ──────────────────────────────────────
    "neutral": [
        "Went to the grocery store and picked up the usual items for the week",
        "Had a regular meeting at work, nothing notable happened",
        "Made coffee and read the morning news as usual",
        "Drove to the office — traffic was about average for a Tuesday",
        "Finished a crossword puzzle while waiting at the dentist",
        "Watched a documentary about whales, it was informative",
        "Did laundry and reorganised the hall closet",
        "Replied to a few emails and scheduled some appointments",
    ],
    "curious": [
        "Started reading about quantum computing and couldn't put the book down",
        "Discovered a new artist whose style is completely unique",
        "Watched a lecture on human memory and it made me rethink everything",
        "Found an interactive map of ancient trade routes and spent hours exploring",
        "Someone mentioned a philosophy I'd never heard of and now I'm deep into it",
        "Took apart an old watch just to see how the mechanism works",
        "Noticed a weird pattern in my sleep data and started researching it",
        "Downloaded a microscope app and spent the evening looking at everyday objects",
    ],
}

# Emotion aliases — map to the primary training emotion
EMOTION_ALIASES: dict[str, str] = {
    "elated": "joyful", "cheerful": "happy", "amused": "happy",
    "enthusiastic": "excited", "thrilled": "excited", "eager": "excited",
    "triumphant": "proud", "confident": "proud", "accomplished": "proud",
    "peaceful": "calm", "serene": "calm", "relaxed": "calm",
    "thankful": "grateful", "appreciative": "grateful",
    "optimistic": "hopeful", "wistful": "nostalgic",
    "affectionate": "loving", "warm": "loving", "tender": "loving",
    "heartbroken": "sad", "grief": "sad", "devastated": "sad",
    "isolated": "lonely", "abandoned": "lonely",
    "ashamed": "guilty", "regretful": "guilty", "embarrassed": "guilty",
    "let_down": "disappointed", "dismayed": "disappointed",
    "irritated": "frustrated", "annoyed": "frustrated",
    "worried": "anxious", "nervous": "anxious", "panicked": "anxious",
    "overwhelmed": "stressed", "exhausted": "stressed",
    "bitter": "angry", "hostile": "angry", "resentful": "angry",
    "enraged": "furious", "livid": "furious",
    "contemplative": "neutral", "indifferent": "neutral",
    "interested": "curious", "fascinated": "curious",
}

# Broader emotion groups (for picking contrasting emotions in triplets)
POSITIVE_HIGH = ["happy", "excited", "proud", "joyful", "loving", "grateful"]
POSITIVE_LOW  = ["calm", "content", "hopeful", "nostalgic"]
NEGATIVE_LOW  = ["sad", "lonely", "melancholy", "disappointed", "guilty"]
NEGATIVE_HIGH = ["angry", "frustrated", "anxious", "stressed", "furious"]
NEUTRAL       = ["neutral", "curious"]

ALL_EMOTIONS  = POSITIVE_HIGH + POSITIVE_LOW + NEGATIVE_LOW + NEGATIVE_HIGH + NEUTRAL

# Valence grouping for contradiction awareness
POSITIVE_EMOTIONS = set(POSITIVE_HIGH + POSITIVE_LOW)
NEGATIVE_EMOTIONS = set(NEGATIVE_LOW + NEGATIVE_HIGH)


# ══════════════════════════════════════════════════════════════════
# 2.  SEMANTIC PARAPHRASE PAIRS
#     Same meaning, different words — preserves core retrieval.
# ══════════════════════════════════════════════════════════════════

SEMANTIC_PAIRS: list[tuple[str, str]] = [
    ("Went jogging in the morning before work", "Did an early morning run before heading to the office"),
    ("Cooked a big dinner for the whole family", "Made a large meal that everyone in the family shared"),
    ("Stayed up late reading my favourite book", "Read well into the night because I couldn't put the book down"),
    ("Had a great conversation with my best friend", "Talked for hours with my closest friend and it was wonderful"),
    ("The weather was beautiful so I went for a walk", "Took a stroll outside because the weather was gorgeous"),
    ("Finished a tough workout at the gym", "Completed an intense exercise session at the gym"),
    ("My dog was so excited to see me come home", "When I got home my dog was absolutely thrilled"),
    ("Spent the weekend decorating the house", "Used the weekend to redecorate the house"),
    ("Got stuck in heavy traffic on the way home", "The commute home was terrible because of heavy traffic"),
    ("Tried a new recipe and it turned out really well", "Made a new dish for the first time and it was delicious"),
    ("Woke up early to watch the sunrise", "Got up at dawn to see the sun come up"),
    ("Played video games with my brother all afternoon", "Spent the whole afternoon gaming with my brother"),
    ("Went to the farmer's market and bought fresh vegetables", "Picked up fresh produce at the local farmer's market"),
    ("Had trouble sleeping last night because of stress", "Couldn't fall asleep because I was too stressed out"),
    ("Started learning a new programming language", "Began teaching myself a new coding language"),
    ("My car broke down on the highway today", "Had a vehicle breakdown on the highway"),
    ("Watched a documentary about climate change", "Saw a film about global warming"),
    ("Met an old friend for coffee after years apart", "Caught up with a friend I hadn't seen in years over coffee"),
    ("Cleaned the entire apartment from top to bottom", "Deep cleaned every room in the apartment"),
    ("The concert last night was absolutely incredible", "Last night's live music show was amazing"),
    ("Took a long bath to relax after a tough day", "Had a relaxing soak in the tub after a hard day"),
    ("My boss praised my work in front of the whole team", "The manager publicly complimented my performance"),
    ("Spent the morning gardening and planting new flowers", "Worked in the garden planting flowers all morning"),
    ("Read a fascinating article about artificial intelligence", "Came across a really interesting piece about AI"),
    ("Made plans to visit my parents next weekend", "Arranged to go see my mum and dad next weekend"),
    ("Couldn't focus on work because my mind kept wandering", "Had trouble concentrating at work, kept getting distracted"),
    ("Went swimming at the lake on a hot summer day", "Took a swim in the lake to cool off in the heat"),
    ("Had an argument with my partner about finances", "Got into a disagreement with my partner over money"),
    ("Baked cookies and brought them to the office", "Made homemade cookies and shared them at work"),
    ("Started a journal to track my thoughts and feelings", "Began keeping a diary to record my inner life"),
    ("Visited the art museum and saw a stunning exhibition", "Went to the gallery and an exhibit blew me away"),
    ("Helped my neighbour carry groceries up the stairs", "Gave my neighbour a hand bringing their shopping upstairs"),
    ("Took a nap in the afternoon and felt refreshed after", "Had an afternoon nap and woke up feeling great"),
    ("Started my mornings with fifteen minutes of meditation", "I've been meditating every morning for a quarter of an hour"),
    ("The power went out during the storm last night", "We lost electricity in last night's storm"),
    ("Adopted a rescue cat from the local shelter", "Got a cat from the animal rescue centre"),
    ("Learned to make sushi from a YouTube tutorial", "Taught myself sushi making by watching YouTube"),
    ("Got into an argument with a stranger on social media", "Had a heated exchange online with someone I don't know"),
    ("Took my kids to the zoo and they loved the elephants", "Brought the children to the zoo — elephants were the highlight"),
    ("Spent the evening stargazing with a telescope", "Used my telescope to look at stars all evening"),
]


# ══════════════════════════════════════════════════════════════════
# 3.  CONTRADICTION PAIRS
#     Opposing sentiments about the same topic — should be FAR.
# ══════════════════════════════════════════════════════════════════

CONTRADICTION_PAIRS: list[tuple[str, str, str]] = [
    # (anchor, agreeing_paraphrase, contradiction)
    ("I absolutely love my job and look forward to Monday mornings",
     "Work is the highlight of my week, I genuinely enjoy what I do",
     "I dread going to work every single day, my job makes me miserable"),
    ("My relationship with my partner has never been stronger",
     "Things between me and my partner are fantastic right now",
     "My relationship feels like it's falling apart and I don't know if we'll make it"),
    ("I feel confident and strong after months of training",
     "My fitness has improved dramatically and I feel powerful",
     "I've never felt weaker, I can barely get through a simple workout"),
    ("Coffee is the best part of my morning routine",
     "I can't start my day properly without a good cup of coffee",
     "I've completely quit coffee and honestly I don't miss it at all"),
    ("I sleep like a baby every night, eight solid hours",
     "My sleep has been excellent lately, deep and uninterrupted",
     "I haven't had a proper night's sleep in weeks, the insomnia is terrible"),
    ("The city is an exciting place to live with so much to do",
     "I love the energy and opportunities of city life",
     "Living in the city is exhausting and isolating, I want to move away"),
    ("I trust my best friend completely with anything",
     "My best friend is the most reliable person I know",
     "I found out my best friend has been lying to me and I can't trust them"),
    ("Cooking is my favourite way to relax and be creative",
     "I find making elaborate meals incredibly satisfying and calming",
     "I hate cooking, it feels like a chore I have to endure every day"),
    ("My mental health has been the best it's ever been",
     "I feel emotionally balanced and mentally clear these days",
     "My mental health is deteriorating and I'm struggling to cope daily"),
    ("Running feels like freedom, I love every mile",
     "Nothing clears my head like a long run through the park",
     "I've grown to hate running, it just feels painful and pointless now"),
    ("My parents are incredibly supportive and understanding",
     "I'm so lucky to have parents who always have my back",
     "My relationship with my parents is toxic and I need distance from them"),
    ("I'm excited about the future and all the possibilities ahead",
     "Looking forward fills me with optimism and energy",
     "The future terrifies me and I feel hopeless about what's coming"),
    ("Moving to this neighbourhood was the best decision I made",
     "I love my neighbourhood, the people are great and it's peaceful",
     "I regret moving here, the area is noisy and the neighbours are hostile"),
    ("Learning guitar has been incredibly rewarding and fun",
     "Playing guitar is one of the most enjoyable things I've picked up",
     "I've given up on guitar, it's frustrating and I'm not making progress"),
    ("This book changed my perspective on everything",
     "Reading this book was a truly transformative experience",
     "The book everyone recommended was a complete waste of my time"),
    ("I feel grateful to have such a strong support system",
     "The people around me make me feel safe and valued",
     "I feel completely alone, nobody in my life truly understands me"),
    ("Working from home has been amazing for my productivity",
     "Remote work lets me focus better than I ever could in the office",
     "Working from home has destroyed my productivity and mental health"),
    ("The holiday trip was the best vacation we've ever taken",
     "Our holiday was absolutely perfect from start to finish",
     "The vacation was a disaster, everything that could go wrong did"),
    ("My new manager is fantastic and really supports the team",
     "Having a great boss has made work enjoyable again",
     "My new manager is terrible and makes every day at work unbearable"),
    ("I'm at peace with who I am and where I'm headed",
     "Self-acceptance has brought me a sense of deep contentment",
     "I constantly feel like I'm not good enough and I'm falling behind"),
]


# ══════════════════════════════════════════════════════════════════
# 4.  MOOD-CONDITIONED QUERY TEMPLATES
#     Queries that should retrieve different memories under
#     different mood states.
# ══════════════════════════════════════════════════════════════════

MOOD_QUERIES: list[str] = [
    "what's been happening lately",
    "tell me about my relationships",
    "how has work been going",
    "what do I do in my free time",
    "recall something important",
    "what have I been feeling recently",
    "something that happened with my family",
    "describe my typical day",
    "anything about my health",
    "what memories stand out",
    "tell me about the past week",
    "what have I been up to",
    "do I have any plans coming up",
    "how have things been going",
    "anything I should remember",
    "what happened this weekend",
    "tell me about my friends",
    "how am I doing overall",
    "what's on my mind",
    "recall a recent experience",
]

# Moods for conditioning queries (subset of emotions)
QUERY_MOODS = [
    "happy", "sad", "angry", "anxious", "calm",
    "excited", "lonely", "hopeful", "nostalgic", "neutral",
]


# ══════════════════════════════════════════════════════════════════
# 5.  GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _get_contrasting_emotions(emo: str) -> list[str]:
    """Return emotions from the opposite valence group."""
    if emo in POSITIVE_EMOTIONS:
        return [e for e in NEGATIVE_EMOTIONS if e in MEMORIES]
    elif emo in NEGATIVE_EMOTIONS:
        return [e for e in POSITIVE_EMOTIONS if e in MEMORIES]
    else:  # neutral
        return [e for e in ALL_EMOTIONS if e != emo and e in MEMORIES]


def _tag_memory(text: str, emotion: str, importance: int | None = None) -> str:
    """Prepend conditioning tokens to a memory."""
    parts = [f"[EMO:{emotion}]"]
    if importance is not None:
        parts.append(f"[IMP:{importance}]")
    parts.append(text)
    return " ".join(parts)


def _tag_query(text: str, mood: str) -> str:
    """Prepend mood + query tokens to a retrieval query."""
    return f"[MOOD:{mood}] [QUERY] {text}"


def generate_emotion_triplets(n: int = 4000) -> list[dict]:
    """Anchor + same-emotion positive + different-emotion negative."""
    triplets = []
    emotion_list = [e for e in ALL_EMOTIONS if e in MEMORIES]

    for _ in range(n):
        emo = random.choice(emotion_list)
        contrast_pool = _get_contrasting_emotions(emo)
        if not contrast_pool:
            continue
        neg_emo = random.choice(contrast_pool)

        mems = MEMORIES[emo]
        neg_mems = MEMORIES[neg_emo]
        if len(mems) < 2:
            continue

        anchor_text, pos_text = random.sample(mems, 2)
        neg_text = random.choice(neg_mems)

        # Use emotion tags with random importance
        imp_a = random.randint(3, 9)
        imp_p = random.randint(3, 9)
        imp_n = random.randint(3, 9)

        triplets.append({
            "anchor":   _tag_memory(anchor_text, emo, imp_a),
            "positive": _tag_memory(pos_text,    emo, imp_p),
            "negative": _tag_memory(neg_text, neg_emo, imp_n),
        })

    return triplets


def generate_semantic_pairs(n: int = 2000) -> list[dict]:
    """Paraphrase pairs — no emotion prefix (raw semantic quality)."""
    pairs = []
    base = list(SEMANTIC_PAIRS)  # 40 pairs
    for _ in range(n):
        a, b = random.choice(base)
        # Randomly swap direction
        if random.random() < 0.5:
            a, b = b, a
        pairs.append({"anchor": a, "positive": b})

    # Also create same-topic pairs from memories
    emotion_list = [e for e in ALL_EMOTIONS if e in MEMORIES]
    for _ in range(n // 2):
        emo = random.choice(emotion_list)
        mems = MEMORIES[emo]
        if len(mems) < 2:
            continue
        a, b = random.sample(mems, 2)
        pairs.append({"anchor": a, "positive": b})

    return pairs


def generate_contradiction_triplets(n: int = 1500) -> list[dict]:
    """Anchor + agreeing paraphrase + contradiction (should be far)."""
    triplets = []
    base = list(CONTRADICTION_PAIRS)

    for _ in range(n):
        anchor, agreement, contradiction = random.choice(base)
        triplets.append({
            "anchor":   anchor,
            "positive": agreement,
            "negative": contradiction,
        })

    return triplets


def generate_mood_triplets(n: int = 2500) -> list[dict]:
    """[MOOD:x] query + mood-congruent memory + mood-incongruent memory."""
    triplets = []

    for _ in range(n):
        mood = random.choice(QUERY_MOODS)
        query_text = random.choice(MOOD_QUERIES)

        # Mood-congruent memory: same valence as mood
        if mood in POSITIVE_EMOTIONS:
            congruent_emo = random.choice(
                [e for e in POSITIVE_EMOTIONS if e in MEMORIES]
            )
            incongruent_emo = random.choice(
                [e for e in NEGATIVE_EMOTIONS if e in MEMORIES]
            )
        elif mood in NEGATIVE_EMOTIONS:
            congruent_emo = random.choice(
                [e for e in NEGATIVE_EMOTIONS if e in MEMORIES]
            )
            incongruent_emo = random.choice(
                [e for e in POSITIVE_EMOTIONS if e in MEMORIES]
            )
        else:  # neutral moods
            congruent_emo = random.choice(
                [e for e in NEUTRAL if e in MEMORIES]
            )
            incongruent_emo = random.choice(
                [e for e in (POSITIVE_HIGH + NEGATIVE_HIGH) if e in MEMORIES]
            )

        pos_text = random.choice(MEMORIES[congruent_emo])
        neg_text = random.choice(MEMORIES[incongruent_emo])

        triplets.append({
            "anchor":   _tag_query(query_text, mood),
            "positive": _tag_memory(pos_text, congruent_emo),
            "negative": _tag_memory(neg_text, incongruent_emo),
        })

    return triplets


# ══════════════════════════════════════════════════════════════════
# 5b. FACTUAL RETRIEVAL PAIRS
#     Natural-language queries → exact stored passages with specific
#     values (tool calls, settings, entities).  Teaches the model to
#     match factual questions to memory passages the way vanilla
#     MiniLM does — keyword / entity overlap — so we DON'T lose
#     factual recall when we add emotional geometry.
# ══════════════════════════════════════════════════════════════════

FACTUAL_RETRIEVAL_TEMPLATES: list[dict] = [
    # ── Tool-call retrieval: query → stored tool usage ──────
    {"query": "check the weather forecast",
     "positive": "Used tool get_weather with args: {\"city\": \"Seattle\", \"units\": \"metric\"}",
     "negative": "User said: I had a great day at the park"},
    {"query": "what stock did I look up",
     "positive": "Used tool get_stock_price with args: {\"symbol\": \"AAPL\"}",
     "negative": "User said: I'm feeling anxious about my finances"},
    {"query": "play that song again",
     "positive": "Used tool play_music with args: {\"track\": \"Bohemian Rhapsody\", \"artist\": \"Queen\"}",
     "negative": "Used tool set_alarm with args: {\"time\": \"07:00\"}"},
    {"query": "send a message to my friend",
     "positive": "Used tool send_message with args: {\"recipient\": \"Alice\", \"text\": \"See you at 7\"}",
     "negative": "User said: I need to buy groceries this weekend"},
    {"query": "set a timer for cooking",
     "positive": "Used tool set_timer with args: {\"duration\": \"25 minutes\", \"label\": \"pasta\"}",
     "negative": "Used tool get_news with args: {\"topic\": \"technology\"}"},
    {"query": "find nearby restaurants",
     "positive": "Used tool search_places with args: {\"query\": \"Italian restaurant\", \"radius\": \"5km\"}",
     "negative": "User said: I went for a jog this morning"},
    {"query": "what email did I draft",
     "positive": "Used tool compose_email with args: {\"to\": \"boss@work.com\", \"subject\": \"Q3 Report\"}",
     "negative": "Used tool play_music with args: {\"track\": \"Stairway to Heaven\"}"},
    {"query": "look up that cryptocurrency",
     "positive": "Used tool get_crypto_price with args: {\"coin\": \"Bitcoin\", \"currency\": \"USD\"}",
     "negative": "User said: I'm saving up for a vacation"},
    {"query": "what did I search for online",
     "positive": "Used tool web_search with args: {\"query\": \"best noise-cancelling headphones 2026\"}",
     "negative": "Used tool get_weather with args: {\"city\": \"London\"}"},
    {"query": "book that flight I mentioned",
     "positive": "Used tool book_flight with args: {\"origin\": \"JFK\", \"destination\": \"LAX\", \"date\": \"2026-04-15\"}",
     "negative": "User said: I need to call the plumber"},
    {"query": "show my calendar for next week",
     "positive": "Used tool get_calendar with args: {\"start\": \"2026-03-23\", \"end\": \"2026-03-29\"}",
     "negative": "Used tool set_reminder with args: {\"text\": \"buy milk\"}"},
    {"query": "translate that phrase to Spanish",
     "positive": "Used tool translate with args: {\"text\": \"Good morning\", \"target_lang\": \"es\"}",
     "negative": "Used tool get_stock_price with args: {\"symbol\": \"TSLA\"}"},
    {"query": "order food delivery",
     "positive": "Used tool order_food with args: {\"restaurant\": \"Luigi's Pizza\", \"items\": [\"margherita\", \"garlic bread\"]}",
     "negative": "User said: I had a quiet evening reading"},
    {"query": "what reminder did I set",
     "positive": "Used tool set_reminder with args: {\"text\": \"Dentist appointment Friday 3pm\", \"time\": \"Friday 14:30\"}",
     "negative": "Used tool translate with args: {\"text\": \"hello\", \"target_lang\": \"fr\"}"},
    {"query": "check my bank balance",
     "positive": "Used tool get_balance with args: {\"account\": \"checking\", \"bank\": \"Chase\"}",
     "negative": "User said: I'm thinking about switching banks"},
    # ── Implicit reference resolution ──────────────────────
    {"query": "that stock we looked at earlier",
     "positive": "Used tool get_stock_price with args: {\"symbol\": \"GOOGL\"}",
     "negative": "User said: The stock market seems volatile lately"},
    {"query": "the restaurant I usually order from",
     "positive": "Used tool order_food with args: {\"restaurant\": \"Panda Express\", \"items\": [\"orange chicken\"]}",
     "negative": "User said: I cooked dinner at home tonight"},
    {"query": "my favourite playlist",
     "positive": "Used tool play_music with args: {\"playlist\": \"Chill Vibes 2026\", \"shuffle\": true}",
     "negative": "Used tool get_news with args: {\"topic\": \"entertainment\"}"},
    {"query": "the person I was emailing about work",
     "positive": "Used tool compose_email with args: {\"to\": \"sarah@company.com\", \"subject\": \"Project Timeline\"}",
     "negative": "Used tool send_message with args: {\"recipient\": \"Mom\", \"text\": \"Happy birthday\"}"},
    {"query": "where was I flying to",
     "positive": "Used tool book_flight with args: {\"origin\": \"SFO\", \"destination\": \"NRT\", \"class\": \"economy\"}",
     "negative": "User said: I love travelling to new places"},
    # ── Entity/value recall ────────────────────────────────
    {"query": "what's the address of the place I searched for",
     "positive": "Used tool get_directions with args: {\"destination\": \"123 Oak Street, Portland\"}",
     "negative": "User said: I need to find a new apartment"},
    {"query": "what temperature did I set the thermostat to",
     "positive": "Used tool set_thermostat with args: {\"temperature\": 72, \"mode\": \"heat\"}",
     "negative": "User said: It's been really cold this week"},
    {"query": "that news article topic",
     "positive": "Used tool get_news with args: {\"topic\": \"artificial intelligence\", \"count\": 5}",
     "negative": "Used tool web_search with args: {\"query\": \"recipe for sourdough bread\"}"},
    {"query": "the Forex pair I asked about",
     "positive": "Used tool get_forex_rate with args: {\"pair\": \"EURUSD\"}",
     "negative": "Used tool get_crypto_price with args: {\"coin\": \"Ethereum\"}"},
    {"query": "check on that social media account",
     "positive": "Used tool get_user_tweets with args: {\"username\": \"elonmusk\", \"limit\": 20}",
     "negative": "User said: I should post more on social media"},
    # ── User-said factual recall ───────────────────────────
    {"query": "what city did I mention I was visiting",
     "positive": "User said: I'm planning a trip to Tokyo next month",
     "negative": "User said: I wish I could travel more often"},
    {"query": "what project was I working on",
     "positive": "User said: I need to finish the Q3 analytics dashboard by Friday",
     "negative": "User said: Work has been really stressful"},
    {"query": "my doctor's name",
     "positive": "User said: Dr. Patel said my test results came back normal",
     "negative": "User said: I had a checkup last week"},
    {"query": "the book I was reading",
     "positive": "User said: I just started reading Project Hail Mary by Andy Weir",
     "negative": "User said: I spent the evening reading"},
    {"query": "what time is the meeting",
     "positive": "User said: The team standup is at 9:30am every Tuesday",
     "negative": "User said: I have too many meetings this week"},
    {"query": "how much did that cost",
     "positive": "Used tool make_purchase with args: {\"item\": \"wireless keyboard\", \"price\": 79.99}",
     "negative": "User said: I need to watch my spending"},
    {"query": "my partner's name",
     "positive": "User said: Alex and I are celebrating our anniversary next week",
     "negative": "User said: We had a lovely dinner together"},
    {"query": "the recipe I looked up",
     "positive": "Used tool web_search with args: {\"query\": \"authentic pad thai recipe\"}",
     "negative": "Used tool web_search with args: {\"query\": \"best running shoes 2026\"}"},
    {"query": "what game was I playing",
     "positive": "User said: I've been obsessed with Elden Ring lately, just beat Malenia",
     "negative": "User said: I spent the afternoon relaxing"},
    {"query": "the medication I mentioned",
     "positive": "User said: Dr. Lee prescribed me 20mg of Lexapro for the anxiety",
     "negative": "User said: I've been feeling better about my health"},
    # ── Cross-domain distractors (same keywords, wrong domain)
    {"query": "show me that video about cooking",
     "positive": "Used tool search_videos with args: {\"query\": \"Gordon Ramsay beef wellington tutorial\"}",
     "negative": "Used tool order_food with args: {\"restaurant\": \"Gordon Ramsay's\"}"},
    {"query": "the alarm I set for work",
     "positive": "Used tool set_alarm with args: {\"time\": \"06:30\", \"label\": \"work\"}",
     "negative": "Used tool set_timer with args: {\"duration\": \"30 minutes\"}"},
    {"query": "what language was I learning",
     "positive": "User said: I started a Japanese course on Duolingo this week",
     "negative": "Used tool translate with args: {\"target_lang\": \"ja\"}"},
    {"query": "the workout routine I asked about",
     "positive": "Used tool web_search with args: {\"query\": \"5x5 stronglifts beginner program\"}",
     "negative": "User said: I went to the gym today"},
]


def generate_factual_pairs(n: int = 3000) -> list[dict]:
    """Generate factual retrieval triplets from templates.

    Uses the FACTUAL_RETRIEVAL_TEMPLATES as seeds and augments with:
      - Random [EMO:neutral] [IMP:n] prefixes on stored passages
        (simulates how VividEmbed stores data)
      - [MOOD:neutral] [QUERY] prefix on queries
        (simulates how VividEmbed queries)
      - Swapped query/positive direction variants
    """
    triplets = []
    templates = list(FACTUAL_RETRIEVAL_TEMPLATES)

    for _ in range(n):
        tmpl = random.choice(templates)
        query = tmpl["query"]
        positive = tmpl["positive"]
        negative = tmpl["negative"]

        # --- Variant A: raw text (backward-compatible with no-prefix queries)
        triplets.append({
            "anchor": query,
            "positive": positive,
            "negative": negative,
        })

        # --- Variant B: with vivid conditioning tokens
        #     (this is the real use-case: VividEmbed stores with tags,
        #      and queries come with [MOOD:] [QUERY] prefix)
        imp_p = random.randint(5, 8)
        imp_n = random.randint(3, 7)
        triplets.append({
            "anchor":   f"[MOOD:neutral] [QUERY] {query}",
            "positive": f"[EMO:neutral] [IMP:{imp_p}] {positive}",
            "negative": f"[EMO:neutral] [IMP:{imp_n}] {negative}",
        })

    return triplets


def generate_magnitude_examples(n: int = 2500) -> list[dict]:
    """Memory text with [IMP:n] tag → target magnitude (n/10)."""
    examples = []
    emotion_list = [e for e in ALL_EMOTIONS if e in MEMORIES]

    for _ in range(n):
        emo = random.choice(emotion_list)
        text = random.choice(MEMORIES[emo])
        importance = random.randint(1, 10)

        examples.append({
            "text":   _tag_memory(text, emo, importance),
            # Target norm mapped to natural MiniLM range (~2-8):
            #   IMP:1 → 2.5,  IMP:5 → 4.5,  IMP:10 → 7.0
            "label":  2.0 + importance * 0.5,
        })

    return examples


# ══════════════════════════════════════════════════════════════════
# 6.  SAVE
# ══════════════════════════════════════════════════════════════════

def build_and_save():
    """Generate all datasets and save to disk."""
    from datasets import Dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating training data for VividEmbedder fine-tuning...")
    print(f"Output: {OUTPUT_DIR}\n")

    # ── Emotion congruence ──────────────────────────────────
    emo_data = generate_emotion_triplets(4000)
    ds_emo = Dataset.from_list(emo_data)
    ds_emo.save_to_disk(str(OUTPUT_DIR / "emotion_triplets"))
    print(f"  Emotion triplets:        {len(emo_data):,}")

    # ── Semantic pairs ──────────────────────────────────────
    sem_data = generate_semantic_pairs(2000)
    ds_sem = Dataset.from_list(sem_data)
    ds_sem.save_to_disk(str(OUTPUT_DIR / "semantic_pairs"))
    print(f"  Semantic pairs:          {len(sem_data):,}")

    # ── Contradiction triplets ──────────────────────────────
    con_data = generate_contradiction_triplets(1500)
    ds_con = Dataset.from_list(con_data)
    ds_con.save_to_disk(str(OUTPUT_DIR / "contradiction_triplets"))
    print(f"  Contradiction triplets:  {len(con_data):,}")

    # ── Mood-conditioned retrieval ──────────────────────────
    mood_data = generate_mood_triplets(2500)
    ds_mood = Dataset.from_list(mood_data)
    ds_mood.save_to_disk(str(OUTPUT_DIR / "mood_triplets"))
    print(f"  Mood retrieval triplets: {len(mood_data):,}")

    # ── Factual retrieval triplets ───────────────────────────
    fact_data = generate_factual_pairs(3000)
    ds_fact = Dataset.from_list(fact_data)
    ds_fact.save_to_disk(str(OUTPUT_DIR / "factual_triplets"))
    print(f"  Factual retrieval trips: {len(fact_data):,}")

    # ── Magnitude examples ──────────────────────────────────
    mag_data = generate_magnitude_examples(2500)
    ds_mag = Dataset.from_list(mag_data)
    ds_mag.save_to_disk(str(OUTPUT_DIR / "magnitude_examples"))
    print(f"  Magnitude examples:      {len(mag_data):,}")

    total = len(emo_data) + len(sem_data) + len(con_data) + len(mood_data) + len(fact_data) + len(mag_data)
    print(f"\n  Total examples:          {total:,}")

    # Also save a JSON manifest
    manifest = {
        "emotion_triplets": len(emo_data),
        "semantic_pairs": len(sem_data),
        "contradiction_triplets": len(con_data),
        "mood_triplets": len(mood_data),
        "factual_triplets": len(fact_data),
        "magnitude_examples": len(mag_data),
        "total": total,
        "special_tokens": [
            f"[EMO:{e}]" for e in ALL_EMOTIONS
        ] + [
            f"[MOOD:{m}]" for m in QUERY_MOODS
        ] + [
            f"[IMP:{i}]" for i in range(1, 11)
        ] + ["[QUERY]"],
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest: {OUTPUT_DIR / 'manifest.json'}")
    print("  Done!")


if __name__ == "__main__":
    build_and_save()
