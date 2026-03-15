"""Generate a synthetic LoCoMo-style benchmark dataset for Fraction evaluation.

Creates multi-turn conversations with ground-truth QA pairs across 4 categories:
1. Single-hop (direct fact recall)
2. Multi-hop (connecting multiple facts)
3. Temporal (time-based reasoning)
4. Open-domain (general knowledge from conversation)
"""

import json
import os

CONVERSATIONS = {
    "conv_1": {
        "conversation": [
            {"role": "Emma", "content": "I just got back from my trip to Tokyo last week. The cherry blossoms were absolutely stunning in Ueno Park."},
            {"role": "Liam", "content": "That sounds amazing! I've always wanted to visit Japan. Did you try any local food?"},
            {"role": "Emma", "content": "Oh yes! I had the best ramen at a tiny shop in Shibuya called Fuunji. The tsukemen was incredible."},
            {"role": "Liam", "content": "I'm actually planning a trip to Seoul next month for a tech conference. Maybe I'll add Tokyo as a stopover."},
            {"role": "Emma", "content": "You should! By the way, I started learning Japanese on Duolingo three months ago. It really helped with reading menus and signs."},
            {"role": "Liam", "content": "That's smart. I've been learning Korean for about six months now using the Talk To Me In Korean course."},
            {"role": "Emma", "content": "How's that going? I heard Korean grammar is quite different from English."},
            {"role": "Liam", "content": "It's challenging but fun. I can hold basic conversations now. My Korean tutor, Mrs. Park, is really patient."},
            {"role": "Emma", "content": "That reminds me, my Japanese teacher Tanaka-sensei recommended visiting Kyoto for the temples. I'm planning to go back in October."},
            {"role": "Liam", "content": "October would be perfect for fall colors! I went to Kyoto two years ago in November and the maple trees were breathtaking at Kiyomizu-dera."},
            {"role": "Emma", "content": "I also picked up some matcha from a tea shop in Asakusa. The owner, Mr. Suzuki, has been making tea for 40 years."},
            {"role": "Liam", "content": "That's dedication. Speaking of dedication, I finally finished my machine learning certification from Stanford last Tuesday."},
            {"role": "Emma", "content": "Congratulations! I remember you started that course last January. Was it worth it?"},
            {"role": "Liam", "content": "Absolutely. The deep learning module by Professor Andrew Ng was the highlight. I'm now applying it to my work at DataFlow Inc."},
            {"role": "Emma", "content": "I'm actually considering switching to a data science role. Currently I'm a product manager at TechVista but I want something more technical."},
            {"role": "Liam", "content": "You'd be great at it! Your analytical skills from PM work would transfer well. Have you looked at any bootcamps?"},
            {"role": "Emma", "content": "I've been eyeing the Springboard data science bootcamp. It's 6 months and they have a job guarantee."},
            {"role": "Liam", "content": "My colleague Sarah did that one and loved it. She's now a senior data analyst at Netflix."},
            {"role": "Emma", "content": "That's encouraging! On a different note, my dog Bailey had his vet checkup yesterday. He's a 3-year-old golden retriever and perfectly healthy."},
            {"role": "Liam", "content": "Glad to hear Bailey is doing well! My cat Whiskers just turned 5. She's been a bit grumpy since I rearranged the living room furniture."},
        ],
        "questions": [
            {"question": "What restaurant did Emma eat ramen at in Tokyo?", "answer": "Fuunji in Shibuya", "category": "1"},
            {"question": "What language learning platform does Emma use?", "answer": "Duolingo", "category": "1"},
            {"question": "Where does Liam work?", "answer": "DataFlow Inc", "category": "1"},
            {"question": "What is the name of Liam's Korean tutor?", "answer": "Mrs. Park", "category": "1"},
            {"question": "What certification did Liam recently complete and from where?", "answer": "Machine learning certification from Stanford", "category": "2"},
            {"question": "What are both Emma and Liam learning, and what resources are they using?", "answer": "Emma is learning Japanese on Duolingo, Liam is learning Korean using Talk To Me In Korean", "category": "2"},
            {"question": "What pets do Emma and Liam have, and what are their names?", "answer": "Emma has a golden retriever named Bailey, Liam has a cat named Whiskers", "category": "2"},
            {"question": "When did Emma visit Tokyo?", "answer": "Last week", "category": "3"},
            {"question": "When is Liam's tech conference in Seoul?", "answer": "Next month", "category": "3"},
            {"question": "When did Liam visit Kyoto?", "answer": "Two years ago in November", "category": "3"},
            {"question": "What career change is Emma considering?", "answer": "Switching from product manager to data science", "category": "4"},
            {"question": "What bootcamp is Emma interested in?", "answer": "Springboard data science bootcamp", "category": "4"},
        ]
    },
    "conv_2": {
        "conversation": [
            {"role": "Sofia", "content": "I just moved to Portland, Oregon last month. The food scene here is incredible."},
            {"role": "Marcus", "content": "Portland is great! I lived there for two years before moving to Austin. Do you miss your old city?"},
            {"role": "Sofia", "content": "A bit. I was in Denver for five years. But the rain in Portland doesn't bother me - it reminds me of growing up in Seattle."},
            {"role": "Marcus", "content": "Seattle kid! That explains the rain tolerance. I grew up in Phoenix, so I'm all about the sunshine here in Austin."},
            {"role": "Sofia", "content": "Ha! Very different climates. So what do you do in Austin? I remember you mentioned something about music production."},
            {"role": "Marcus", "content": "Yeah, I run a small recording studio called Sunset Sounds. We mostly work with indie bands and singer-songwriters."},
            {"role": "Sofia", "content": "That's so cool. I'm a freelance graphic designer. Most of my clients are tech startups. My biggest client right now is a fintech company called PayBridge."},
            {"role": "Marcus", "content": "Nice! My sister actually works in fintech at Square. Design is such a crucial part of those products."},
            {"role": "Sofia", "content": "Absolutely. I just finished a complete rebrand for PayBridge - new logo, color palette, the whole thing. Took about three months."},
            {"role": "Marcus", "content": "That's a big project. Speaking of big projects, I'm recording an album for a band called The Midnight Owls. They're incredible - kind of indie folk meets electronic."},
            {"role": "Sofia", "content": "I love that genre blend! Have you worked with them before?"},
            {"role": "Marcus", "content": "This is our second album together. The first one, 'Echoes in the Rain', got 2 million streams on Spotify."},
            {"role": "Sofia", "content": "Impressive! By the way, I've been getting into pottery lately. I take classes at a studio called Clay & Fire every Saturday morning."},
            {"role": "Marcus", "content": "That's a great hobby. I've been really into rock climbing. I go to the Austin Rock Gym three times a week."},
            {"role": "Sofia", "content": "Staying active is important. I also run - I'm training for the Portland Marathon in October. This will be my third marathon."},
            {"role": "Marcus", "content": "Respect! The most I can handle is a 10K. My wife Elena runs marathons though - she did Boston last year."},
            {"role": "Sofia", "content": "Boston is the dream! What was her time?"},
            {"role": "Marcus", "content": "3 hours 22 minutes. She's been running since college at UT Austin. She's actually a physical therapist at Austin Sports Medicine."},
            {"role": "Sofia", "content": "That's an amazing time. My personal best is 3:45 from the Chicago Marathon two years ago."},
            {"role": "Marcus", "content": "Still very impressive. Hey, are you going to the Portland Design Week in November? I heard the keynote speaker is Jessica Walsh."},
        ],
        "questions": [
            {"question": "What is the name of Marcus's recording studio?", "answer": "Sunset Sounds", "category": "1"},
            {"question": "What is Sofia's biggest client?", "answer": "PayBridge, a fintech company", "category": "1"},
            {"question": "What band is Marcus currently recording an album for?", "answer": "The Midnight Owls", "category": "1"},
            {"question": "Where does Sofia take pottery classes?", "answer": "Clay & Fire", "category": "1"},
            {"question": "What cities have both Sofia and Marcus lived in, and where are they now?", "answer": "Sofia lived in Seattle, Denver, now Portland. Marcus lived in Portland, now Austin.", "category": "2"},
            {"question": "What physical activities do Sofia and Marcus do?", "answer": "Sofia runs marathons and does pottery. Marcus does rock climbing.", "category": "2"},
            {"question": "What marathon-related achievements do Sofia and Marcus's wife Elena have?", "answer": "Sofia's PB is 3:45 from Chicago. Elena ran Boston in 3:22.", "category": "2"},
            {"question": "When did Sofia move to Portland?", "answer": "Last month", "category": "3"},
            {"question": "How long did the PayBridge rebrand take?", "answer": "About three months", "category": "3"},
            {"question": "When is the Portland Marathon?", "answer": "October", "category": "3"},
            {"question": "What genre of music does The Midnight Owls play?", "answer": "Indie folk meets electronic", "category": "4"},
            {"question": "What is Elena's profession?", "answer": "Physical therapist at Austin Sports Medicine", "category": "4"},
        ]
    },
    "conv_3": {
        "conversation": [
            {"role": "Aisha", "content": "I just got promoted to senior architect at Foster & Webb. I've been there for 7 years now."},
            {"role": "Noah", "content": "Congratulations! That's a big deal. I remember when you first joined as a junior. What projects are you leading?"},
            {"role": "Aisha", "content": "I'm heading the design for the new Central Library downtown. It's a $50 million project with a 2-year timeline."},
            {"role": "Noah", "content": "That's massive. I'm working on something smaller but exciting - a sustainable housing development in the suburbs called Green Haven."},
            {"role": "Aisha", "content": "Oh I read about Green Haven! It's the one with solar panels on every unit and the community garden, right?"},
            {"role": "Noah", "content": "Exactly! 120 units, all net-zero energy. We broke ground last March and should be done by next summer."},
            {"role": "Aisha", "content": "The sustainability angle is so important. For the library, we're using reclaimed wood and a living green roof. The structural engineer, Dr. Chen, came up with an innovative bamboo framework."},
            {"role": "Noah", "content": "Dr. Chen is brilliant. She consulted on our project too. Her bamboo composite research at MIT was groundbreaking."},
            {"role": "Aisha", "content": "Small world! Speaking of school, my daughter Zara just started first grade at Oakwood Elementary. She's loving it."},
            {"role": "Noah", "content": "That's great! My twins, Leo and Maya, are in third grade at Riverside Academy. They just won their school science fair with a project about water filtration."},
            {"role": "Aisha", "content": "That's adorable and impressive! Zara is really into dinosaurs right now. She wants to be a paleontologist."},
            {"role": "Noah", "content": "Ha! Leo wants to be an astronaut and Maya wants to be a marine biologist. Kids dream big."},
            {"role": "Aisha", "content": "As they should. By the way, have you tried that new Ethiopian restaurant on 5th Street? Addis Kitchen? Their injera is the best I've had outside of Addis Ababa."},
            {"role": "Noah", "content": "Not yet, but I've heard great things. I've been cooking a lot at home lately. I took a Thai cooking class at the community center last month."},
            {"role": "Aisha", "content": "That sounds fun! I've been meaning to take their Italian cooking class. My husband Omar is obsessed with making fresh pasta."},
            {"role": "Noah", "content": "Fresh pasta is a game changer. By the way, I'm presenting our Green Haven project at the National Architecture Conference in Chicago in February."},
            {"role": "Aisha", "content": "I'll be there too! I'm on a panel about public space design. We should grab dinner while we're there."},
            {"role": "Noah", "content": "Definitely! I know a great deep-dish place called Lou Malnati's. It's a Chicago institution."},
            {"role": "Aisha", "content": "Perfect. Also, I wanted to ask - are you still volunteering with Habitat for Humanity on weekends?"},
            {"role": "Noah", "content": "Yes! We just finished building our 15th house last weekend. It's incredibly rewarding. The family, the Garcias, were so grateful."},
        ],
        "questions": [
            {"question": "What project is Aisha leading?", "answer": "The new Central Library downtown", "category": "1"},
            {"question": "How many units does Green Haven have?", "answer": "120 units", "category": "1"},
            {"question": "Who is the structural engineer working on the library?", "answer": "Dr. Chen", "category": "1"},
            {"question": "What are Noah's twins' names?", "answer": "Leo and Maya", "category": "1"},
            {"question": "What do both Aisha and Noah have in common professionally?", "answer": "Both are architects, and Dr. Chen consulted on both their projects", "category": "2"},
            {"question": "What do Noah's twins want to be when they grow up?", "answer": "Leo wants to be an astronaut, Maya wants to be a marine biologist", "category": "2"},
            {"question": "What are both architects presenting at the Chicago conference?", "answer": "Noah is presenting Green Haven, Aisha is on a panel about public space design", "category": "2"},
            {"question": "When did Green Haven break ground?", "answer": "Last March", "category": "3"},
            {"question": "When is the National Architecture Conference?", "answer": "February in Chicago", "category": "3"},
            {"question": "How long has Aisha been at Foster & Webb?", "answer": "7 years", "category": "3"},
            {"question": "What volunteer work does Noah do?", "answer": "Volunteers with Habitat for Humanity building houses", "category": "4"},
            {"question": "What is the budget for the Central Library project?", "answer": "$50 million", "category": "4"},
        ]
    },
    "conv_4": {
        "conversation": [
            {"role": "Priya", "content": "I just submitted my PhD thesis on quantum computing error correction at Cambridge. The defense is scheduled for next month."},
            {"role": "James", "content": "That's incredible, Priya! Five years of work coming to a head. How are you feeling?"},
            {"role": "Priya", "content": "Nervous but excited. My advisor Professor Williams has been really supportive. He thinks the thesis is strong."},
            {"role": "James", "content": "It should be - your paper on topological qubits got published in Nature last year. That's a huge deal."},
            {"role": "Priya", "content": "Thanks! That paper was a collaboration with the IBM Quantum team. Dr. Sarah Martinez led their side of the research."},
            {"role": "James", "content": "I actually met Dr. Martinez at a conference in Zurich last spring. She gave a brilliant talk on superconducting circuits."},
            {"role": "Priya", "content": "Small world! So what about you? How's the startup going?"},
            {"role": "James", "content": "BioSynth is doing well! We just closed our Series A - $8 million from Andreessen Horowitz. We're building AI models for drug discovery."},
            {"role": "Priya", "content": "That's amazing funding. How many people on the team now?"},
            {"role": "James", "content": "We're up to 25. Just hired a CTO, David Park, who was previously at DeepMind working on AlphaFold."},
            {"role": "Priya", "content": "AlphaFold is legendary. Having that expertise will be invaluable. What's your first drug target?"},
            {"role": "James", "content": "We're starting with Alzheimer's. Our AI identified 3 novel protein targets in the first 6 months. We're now in pre-clinical trials with Pfizer."},
            {"role": "Priya", "content": "Partnerships with big pharma already - that's fast. Speaking of health, I've been dealing with some RSI from all the typing. My physiotherapist recommended ergonomic changes."},
            {"role": "James", "content": "Take care of yourself! I switched to a standing desk and split keyboard last year. Made a huge difference for my wrists."},
            {"role": "Priya", "content": "I might try that. On a lighter note, I've been binge-watching 'The Expanse'. Have you seen it?"},
            {"role": "James", "content": "Love it! I'm actually more into reading sci-fi. Just finished 'Project Hail Mary' by Andy Weir. Couldn't put it down."},
            {"role": "Priya", "content": "That's on my list! My book club just finished 'Klara and the Sun' by Kazuo Ishiguro. Very thought-provoking AI themes."},
            {"role": "James", "content": "Ishiguro is a master. Hey, are you planning to stay in academia after your PhD?"},
            {"role": "Priya", "content": "I have a postdoc offer at MIT in Professor Lisa Chen's quantum information lab. I'd be working on quantum machine learning."},
            {"role": "James", "content": "MIT! That's the dream. We should collaborate - quantum ML for drug discovery could be revolutionary."},
        ],
        "questions": [
            {"question": "What is Priya's PhD thesis about?", "answer": "Quantum computing error correction", "category": "1"},
            {"question": "How much funding did BioSynth raise in Series A?", "answer": "$8 million from Andreessen Horowitz", "category": "1"},
            {"question": "Who is BioSynth's new CTO?", "answer": "David Park, previously at DeepMind", "category": "1"},
            {"question": "What book did James recently finish?", "answer": "Project Hail Mary by Andy Weir", "category": "1"},
            {"question": "What connects Dr. Sarah Martinez to both Priya and James?", "answer": "She collaborated with Priya on the Nature paper and James met her at a conference in Zurich", "category": "2"},
            {"question": "What are Priya's and James's next career steps?", "answer": "Priya has a postdoc at MIT, James is growing BioSynth with Pfizer partnership", "category": "2"},
            {"question": "What sci-fi do Priya and James enjoy?", "answer": "Priya watches The Expanse and reads Klara and the Sun, James reads Project Hail Mary", "category": "2"},
            {"question": "When was Priya's Nature paper published?", "answer": "Last year", "category": "3"},
            {"question": "When is Priya's thesis defense?", "answer": "Next month", "category": "3"},
            {"question": "How long has Priya been working on her PhD?", "answer": "Five years", "category": "3"},
            {"question": "What disease is BioSynth targeting first?", "answer": "Alzheimer's", "category": "4"},
            {"question": "What field would a Priya-James collaboration focus on?", "answer": "Quantum machine learning for drug discovery", "category": "4"},
        ]
    },
    "conv_5": {
        "conversation": [
            {"role": "Chen", "content": "I finally finished renovating the kitchen. It took four months and cost way more than I budgeted - $35,000 instead of $25,000."},
            {"role": "Olivia", "content": "Renovations always go over budget! But I bet it looks amazing. What style did you go with?"},
            {"role": "Chen", "content": "Modern minimalist with a Scandinavian touch. White oak cabinets, quartz countertops, and a gorgeous backsplash from a local artisan in Vermont named Martha Cole."},
            {"role": "Olivia", "content": "That sounds beautiful. I'm actually in the middle of converting my garage into an art studio. My contractor, Alejandro Ruiz, says it'll be done by December."},
            {"role": "Chen", "content": "An art studio! That's perfect for you. Are you still doing watercolors or have you branched out?"},
            {"role": "Olivia", "content": "I've gotten really into oil painting this year. I sold three pieces at the Riverside Art Fair in September for $500 each."},
            {"role": "Chen", "content": "That's fantastic! You're becoming a real professional artist. My creative outlet is more digital - I've been learning 3D modeling in Blender."},
            {"role": "Olivia", "content": "Blender is so powerful. What are you making?"},
            {"role": "Chen", "content": "Architectural visualizations mostly, for my real estate business. Speaking of which, I just listed a Victorian house on Maple Street for $1.2 million."},
            {"role": "Olivia", "content": "A Victorian on Maple? That's the one with the turret, right? I drive past it every day on my way to the school where I teach."},
            {"role": "Chen", "content": "That's the one! The sellers, the Thompson family, have been there for 30 years. It needs some updating but the bones are incredible."},
            {"role": "Olivia", "content": "I can imagine. By the way, how's your mother doing? I remember she had hip surgery in August."},
            {"role": "Chen", "content": "She's recovering well, thanks for asking. She's doing physical therapy twice a week at the Greenfield Rehab Center. The therapist, Mike Torres, is excellent."},
            {"role": "Olivia", "content": "Glad to hear it. My dad had the same surgery two years ago and the recovery was smooth. He's back to playing golf every weekend."},
            {"role": "Chen", "content": "That's encouraging. Mom is determined to get back to her tai chi classes. She's been practicing for 20 years!"},
            {"role": "Olivia", "content": "20 years of tai chi - that's dedication. I've been trying to get into yoga. I go to a studio called Namaste Flow on Thursday evenings."},
            {"role": "Chen", "content": "Yoga is great for stress. I manage stress through cooking. Just made a 12-hour beef brisket last weekend for a neighborhood barbecue."},
            {"role": "Olivia", "content": "You're quite the chef! I'm hopeless in the kitchen. My husband David does all the cooking in our house."},
            {"role": "Chen", "content": "David's a great cook from what I remember at your dinner party. Is he still working at the hospital?"},
            {"role": "Olivia", "content": "Yes, he's head of cardiology at St. Mary's Hospital now. Got the promotion in June. Very proud of him."},
        ],
        "questions": [
            {"question": "How much did Chen's kitchen renovation cost?", "answer": "$35,000", "category": "1"},
            {"question": "What is the name of Olivia's contractor?", "answer": "Alejandro Ruiz", "category": "1"},
            {"question": "How much did Olivia sell her paintings for?", "answer": "$500 each", "category": "1"},
            {"question": "What is Chen learning in Blender?", "answer": "3D modeling for architectural visualizations", "category": "1"},
            {"question": "What creative activities do both Chen and Olivia pursue?", "answer": "Chen does 3D modeling in Blender, Olivia does oil painting", "category": "2"},
            {"question": "What health-related events have affected both Chen's and Olivia's families?", "answer": "Chen's mother had hip surgery in August, Olivia's dad had hip surgery two years ago", "category": "2"},
            {"question": "What do Chen and Olivia each do for stress relief and wellness?", "answer": "Chen cooks, Olivia does yoga at Namaste Flow", "category": "2"},
            {"question": "When was the Riverside Art Fair?", "answer": "September", "category": "3"},
            {"question": "When will Olivia's garage studio be done?", "answer": "December", "category": "3"},
            {"question": "When did David get his promotion?", "answer": "June", "category": "3"},
            {"question": "What is David's role at the hospital?", "answer": "Head of cardiology at St. Mary's Hospital", "category": "4"},
            {"question": "What Victorian house did Chen list and for how much?", "answer": "Victorian on Maple Street for $1.2 million", "category": "4"},
        ]
    },
}


def generate_dataset(output_path: str = "benchmarks/dataset/locomo10.json"):
    """Generate and save the benchmark dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(CONVERSATIONS, f, indent=2)

    # Count stats
    total_q = sum(len(c["questions"]) for c in CONVERSATIONS.values())
    total_turns = sum(len(c["conversation"]) for c in CONVERSATIONS.values())
    cats = {}
    for c in CONVERSATIONS.values():
        for q in c["questions"]:
            cat = q["category"]
            cats[cat] = cats.get(cat, 0) + 1

    print(f"Generated dataset: {output_path}")
    print(f"  Conversations: {len(CONVERSATIONS)}")
    print(f"  Total turns: {total_turns}")
    print(f"  Total questions: {total_q}")
    print(f"  By category: {cats}")


if __name__ == "__main__":
    generate_dataset()
