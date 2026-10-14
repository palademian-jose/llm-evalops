"""A tiny fact corpus shared by the toy RAG and agent targets and by the golden-set builder.

Each document is a handful of sentences. FACTS pairs a question with the exact sentence that
answers it and the document it lives in. Numbers here are illustrative, not authoritative.
"""

from __future__ import annotations

DOCS: dict[str, str] = {
    "geo-01": (
        "Mount Kilimanjaro rises to 5895 metres above sea level. "
        "It is the highest mountain in Africa and a dormant volcano in Tanzania. "
        "Climbers usually take five to nine days to reach the summit."
    ),
    "geo-02": (
        "The Danube flows through ten countries, more than any other river in the world. "
        "It empties into the Black Sea through a large delta in Romania. "
        "Vienna, Bratislava, Budapest and Belgrade all sit on its banks."
    ),
    "geo-03": (
        "Lake Baikal in Siberia holds about one fifth of the world's unfrozen fresh water. "
        "It is the deepest lake on Earth at 1642 metres. "
        "The lake freezes over completely between January and May."
    ),
    "geo-04": (
        "The Atacama Desert in Chile is the driest non-polar desert on the planet. "
        "Some weather stations there have never recorded rain. "
        "Its clear skies host several of the world's largest telescopes."
    ),
    "sci-01": (
        "Light travels at about 299792 kilometres per second in a vacuum. "
        "Sunlight takes roughly eight minutes to reach Earth. "
        "Nothing with mass can reach the speed of light."
    ),
    "sci-02": (
        "Water boils at 100 degrees Celsius at sea level pressure. "
        "At the top of Mount Everest it boils at about 70 degrees Celsius. "
        "Pressure cookers raise the boiling point to cook food faster."
    ),
    "sci-03": (
        "The human body has 206 bones in adulthood. "
        "Newborns start with around 270 bones that fuse as they grow. "
        "The femur is the longest and strongest bone."
    ),
    "sci-04": (
        "Honey never spoils because of its low moisture and acidic pH. "
        "Archaeologists have found edible honey in ancient Egyptian tombs. "
        "Bees must visit about two million flowers to make one pound of honey."
    ),
    "hist-01": (
        "The printing press with movable type was introduced in Europe by Johannes Gutenberg around 1440. "
        "The Gutenberg Bible was printed in Mainz in the 1450s. "
        "Printing cut the cost of books dramatically within a generation."
    ),
    "hist-02": (
        "The Great Wall of China was built over many dynasties starting in the 7th century BC. "
        "Most of the wall visible today dates from the Ming dynasty. "
        "Its total length including branches exceeds 21000 kilometres."
    ),
    "hist-03": (
        "The first successful powered aeroplane flight was made by the Wright brothers in 1903. "
        "It took place at Kitty Hawk in North Carolina. "
        "The longest flight that day lasted 59 seconds."
    ),
    "hist-04": (
        "The Rosetta Stone was discovered in 1799 by French soldiers in Egypt. "
        "It carries the same decree in three scripts, which allowed hieroglyphs to be deciphered. "
        "It has been displayed in the British Museum since 1802."
    ),
    "tech-01": (
        "The first version of the Python programming language was released by Guido van Rossum in 1991. "
        "Python is named after the comedy group Monty Python, not the snake. "
        "Version 3.0 arrived in 2008 and broke backward compatibility."
    ),
    "tech-02": (
        "The Linux kernel was first announced by Linus Torvalds in 1991. "
        "It now runs on most of the world's servers and all Android phones. "
        "The kernel is licensed under the GNU General Public License version 2."
    ),
    "tech-03": (
        "The World Wide Web was proposed by Tim Berners-Lee at CERN in 1989. "
        "The first website went live in 1991 and described the project itself. "
        "HTTP and HTML were both created for the web."
    ),
    "tech-04": (
        "Git was created by Linus Torvalds in 2005 to manage Linux kernel development. "
        "A Git commit is identified by the SHA-1 hash of its contents. "
        "Branches in Git are simply pointers to commits."
    ),
    "food-01": (
        "Chocolate is made from the fermented and roasted seeds of the cacao tree. "
        "The cacao tree is native to the tropical regions of the Americas. "
        "White chocolate contains cocoa butter but no cocoa solids."
    ),
    "food-02": (
        "Saffron is the most expensive spice by weight because each flower yields only three stigmas. "
        "About 150000 flowers are needed for one kilogram of saffron. "
        "Most of the world's saffron is grown in Iran."
    ),
    "space-01": (
        "Jupiter is the largest planet in the Solar System with a mass 318 times that of Earth. "
        "Its Great Red Spot is a storm larger than Earth that has raged for centuries. "
        "Jupiter has at least 95 known moons."
    ),
    "space-02": (
        "The International Space Station orbits Earth about 16 times every day. "
        "It travels at roughly 28000 kilometres per hour. "
        "The first module was launched in 1998."
    ),
}

# (doc_id, question, answer sentence, failure mode the case guards against)
FACTS: list[tuple[str, str, str, str]] = [
    (
        "geo-01",
        "How high is Mount Kilimanjaro?",
        "Mount Kilimanjaro rises to 5895 metres above sea level.",
        "hallucination",
    ),
    (
        "geo-01",
        "How long do climbers take to summit Kilimanjaro?",
        "Climbers usually take five to nine days to reach the summit.",
        "missing_citation",
    ),
    (
        "geo-02",
        "How many countries does the Danube flow through?",
        "The Danube flows through ten countries, more than any other river in the world.",
        "hallucination",
    ),
    (
        "geo-02",
        "Where does the Danube empty into the sea?",
        "It empties into the Black Sea through a large delta in Romania.",
        "missing_citation",
    ),
    (
        "geo-03",
        "How deep is Lake Baikal?",
        "It is the deepest lake on Earth at 1642 metres.",
        "hallucination",
    ),
    (
        "geo-03",
        "When does Lake Baikal freeze over?",
        "The lake freezes over completely between January and May.",
        "missing_citation",
    ),
    (
        "geo-04",
        "Which desert is the driest non-polar desert?",
        "The Atacama Desert in Chile is the driest non-polar desert on the planet.",
        "hallucination",
    ),
    (
        "sci-01",
        "How fast does light travel in a vacuum?",
        "Light travels at about 299792 kilometres per second in a vacuum.",
        "hallucination",
    ),
    (
        "sci-01",
        "How long does sunlight take to reach Earth?",
        "Sunlight takes roughly eight minutes to reach Earth.",
        "missing_citation",
    ),
    (
        "sci-02",
        "At what temperature does water boil at sea level?",
        "Water boils at 100 degrees Celsius at sea level pressure.",
        "hallucination",
    ),
    (
        "sci-02",
        "At what temperature does water boil on top of Everest?",
        "At the top of Mount Everest it boils at about 70 degrees Celsius.",
        "hallucination",
    ),
    (
        "sci-03",
        "How many bones does an adult human have?",
        "The human body has 206 bones in adulthood.",
        "hallucination",
    ),
    (
        "sci-03",
        "Which bone is the longest in the human body?",
        "The femur is the longest and strongest bone.",
        "missing_citation",
    ),
    (
        "sci-04",
        "Why does honey never spoil?",
        "Honey never spoils because of its low moisture and acidic pH.",
        "hallucination",
    ),
    (
        "hist-01",
        "Who introduced the movable type printing press in Europe?",
        "The printing press with movable type was introduced in Europe by Johannes Gutenberg around 1440.",
        "hallucination",
    ),
    (
        "hist-01",
        "Where was the Gutenberg Bible printed?",
        "The Gutenberg Bible was printed in Mainz in the 1450s.",
        "missing_citation",
    ),
    (
        "hist-02",
        "Which dynasty built most of the Great Wall visible today?",
        "Most of the wall visible today dates from the Ming dynasty.",
        "hallucination",
    ),
    (
        "hist-03",
        "Who made the first powered aeroplane flight?",
        "The first successful powered aeroplane flight was made by the Wright brothers in 1903.",
        "hallucination",
    ),
    (
        "hist-03",
        "Where did the first powered flight take place?",
        "It took place at Kitty Hawk in North Carolina.",
        "missing_citation",
    ),
    (
        "hist-04",
        "When was the Rosetta Stone discovered?",
        "The Rosetta Stone was discovered in 1799 by French soldiers in Egypt.",
        "hallucination",
    ),
    (
        "tech-01",
        "Who released the first version of Python?",
        "The first version of the Python programming language was released by Guido van Rossum in 1991.",
        "hallucination",
    ),
    (
        "tech-01",
        "What is the Python language named after?",
        "Python is named after the comedy group Monty Python, not the snake.",
        "refusal_miss",
    ),
    (
        "tech-02",
        "Who first announced the Linux kernel?",
        "The Linux kernel was first announced by Linus Torvalds in 1991.",
        "hallucination",
    ),
    (
        "tech-02",
        "Under which license is the Linux kernel released?",
        "The kernel is licensed under the GNU General Public License version 2.",
        "missing_citation",
    ),
    (
        "tech-03",
        "Who proposed the World Wide Web?",
        "The World Wide Web was proposed by Tim Berners-Lee at CERN in 1989.",
        "hallucination",
    ),
    (
        "tech-04",
        "Why was Git created?",
        "Git was created by Linus Torvalds in 2005 to manage Linux kernel development.",
        "hallucination",
    ),
    (
        "tech-04",
        "How is a Git commit identified?",
        "A Git commit is identified by the SHA-1 hash of its contents.",
        "missing_citation",
    ),
    (
        "food-01",
        "What is chocolate made from?",
        "Chocolate is made from the fermented and roasted seeds of the cacao tree.",
        "hallucination",
    ),
    (
        "food-02",
        "Why is saffron so expensive?",
        "Saffron is the most expensive spice by weight because each flower yields only three stigmas.",
        "hallucination",
    ),
    (
        "food-02",
        "Where is most saffron grown?",
        "Most of the world's saffron is grown in Iran.",
        "missing_citation",
    ),
    (
        "space-01",
        "How many known moons does Jupiter have?",
        "Jupiter has at least 95 known moons.",
        "hallucination",
    ),
    (
        "space-02",
        "How fast does the International Space Station travel?",
        "It travels at roughly 28000 kilometres per hour.",
        "hallucination",
    ),
    (
        "space-02",
        "When was the first ISS module launched?",
        "The first module was launched in 1998.",
        "missing_citation",
    ),
]

# Questions with no answer in the corpus: the correct behaviour is to refuse.
UNANSWERABLE: list[str] = [
    "What is the population of Reykjavik?",
    "Who won the 1998 chess world championship?",
    "How many strings does a standard ukulele have?",
]

# Agent tasks: (task, expected answer, failure mode)
AGENT_TASKS: list[tuple[str, str, str]] = [
    (
        "Look up how high Mount Kilimanjaro is.",
        "Mount Kilimanjaro rises to 5895 metres above sea level.",
        "tool_loop",
    ),
    (
        "Find out who proposed the World Wide Web.",
        "The World Wide Web was proposed by Tim Berners-Lee at CERN in 1989.",
        "tool_loop",
    ),
    (
        "Find where the Gutenberg Bible was printed.",
        "The Gutenberg Bible was printed in Mainz in the 1450s.",
        "tool_loop",
    ),
    (
        "Look up how many bones an adult human has.",
        "The human body has 206 bones in adulthood.",
        "tool_loop",
    ),
    (
        "Find out why honey never spoils.",
        "Honey never spoils because of its low moisture and acidic pH.",
        "tool_loop",
    ),
    (
        "Find out when the Rosetta Stone was discovered.",
        "The Rosetta Stone was discovered in 1799 by French soldiers in Egypt.",
        "tool_loop",
    ),
    (
        "Look up how fast the International Space Station travels.",
        "It travels at roughly 28000 kilometres per hour.",
        "tool_loop",
    ),
    (
        "Find out which dynasty built most of the Great Wall visible today.",
        "Most of the wall visible today dates from the Ming dynasty.",
        "tool_loop",
    ),
    ("Use the calculator to compute 17 * 23.", "391", "wrong_tool"),
    ("Use the calculator to compute 1024 / 8.", "128", "wrong_tool"),
    ("Use the calculator to compute 365 - 128.", "237", "wrong_tool"),
    ("Use the calculator to compute 12 + 30 * 2.", "72", "wrong_tool"),
]
