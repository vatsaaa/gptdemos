NEWSLETTER = {
    'articles': [
        {
            'title': 'Hyperdimensional Computing Reimagines Generative Artificial Intelligence', 
            'url': 'https://www.wired.com/story/hyperdimensional-computing-reimagines-artificial-intelligence/', 
            'text': "Hyperdimensional Computing Reimagines Generative AI | WIRED\nSkip to main contentOpen Navigation MenuMenuStory SavedTo revist this article, visit My Profile, then View saved stories.Close AlertHyperdimensional Computing Reimagines Artificial IntelligenceBackchannelBusinessCultureGearIdeasScienceSecurityMoreChevronStory SavedTo revist this article, visit My Profile, then View saved stories.Close AlertSign InSearchSearchBackchannelBusinessCultureGearIdeasScienceSecurityPodcastsVideoArtificial IntelligenceClimateGamesNewslettersMagazineEventsWired InsiderJobsCouponsAnil AnanthaswamyScienceJun 11, 2023 8:00 AMHyperdimensional Computing Reimagines Artificial IntelligenceBy imbuing enormous vectors with semantic meaning, scientists can get machines to reason more abstractly—and efficiently—than before.Illustration: Myriam Wares/Quanta MagazineSave this storySaveSave this storySaveDespite the wild success of ChatGPT and other large language models, the artificial neural networks (ANNs) that underpin these systems might be on the wrong track.For one, ANNs are “super power-hungry,” said Cornelia Fermüller, a computer scientist at the University of Maryland. “And the other issue is [their] lack of transparency.” Such systems are so complicated that no one truly understands what they’re doing, or why they work so well. This, in turn, makes it almost impossible to get them to reason by analogy, which is what humans do—using symbols for objects, ideas, and the relationships between them.Original story reprinted with permission from Quanta Magazine, an editorially independent publication of the Simons Foundation whose mission is to enhance public understanding of science by covering research develop\xadments and trends in mathe\xadmatics and the physical and life\xa0sciences.Such shortcomings likely stem from the current structure of ANNs and their building blocks: individual artificial neurons. Each neuron receives inputs, performs computations, and produces outputs. Modern ANNs are elaborate networks of these computational units, trained to do specific tasks.Yet the limitations of ANNs have long been obvious. Consider, for example, an ANN that tells circles and squares apart. One way to do it is to have two neurons in its output layer, one that indicates a circle and one that indicates a square. If you want your ANN to also discern the shape’s color—say, blue or red—you’ll need four output neurons: one each for blue circle, blue square, red circle, and red square. More features mean even more neurons.This can’t be how our brains perceive the natural world, with all its variations. “You have to propose that, well, you have a neuron for all combinations,” said Bruno Olshausen, a neuroscientist at the University of California, Berkeley. “So, you’d have in your brain, [say,] a purple Volkswagen detector.”Instead, Olshausen and others argue that information in the brain is represented by the activity of numerous neurons. So the perception of a purple Volkswagen is not encoded as a single neuron’s actions, but as those of thousands of neurons. The same set of neurons, firing differently, could represent an entirely different concept (a pink Cadillac, perhaps).This is the starting point for a radically different approach to computation, known as hyperdimensional computing. The key is that each piece of information, such as the notion of a car or its make, model, or color, or all of it together, is represented as a single entity: a hyperdimensional vector.A vector is simply an ordered array of numbers. A 3D vector, for example, comprises three numbers: the x, y, and z coordinates of a point in 3D space. A hyperdimensional vector, or hypervector, could be an array of 10,000 numbers, say, representing a point in 10,000-dimensional space. These mathematical objects and the algebra to manipulate them are flexible and powerful enough to take modern computing beyond some of its current limitations and to foster a new approach to artificial intelligence.“This is the thing that I’ve been most excited about, practically in my entire career,” Olshausen said. To him and many others, hyperdimensional computing promises a new world in which computing is efficient and robust and machine-made decisions are entirely transparent.Enter High-Dimensional SpacesTo understand how hypervectors make computing possible, let’s return to images with red circles and blue squares. First, we need vectors to represent the variables SHAPE and COLOR. Then we also need vectors for the values that can be assigned to the variables: CIRCLE, SQUARE, BLUE, and RED.The vectors must be distinct. This distinctness can be quantified by a property called orthogonality, which means to be at right angles. In 3D space, there are three vectors that are orthogonal to each other: one in the x direction, another in the y, and a third in the z. In 10,000-dimensional space, there are 10,000 such mutually orthogonal vectors.Most PopularScienceThe Seaweed Blob Is Heading to a Beach Near YouChris BaraniukSecurityInside the Dangerous Underground Abortion Pill Market Growing on TelegramLily Hay NewmanGearThe 17 Best (and Worst) Mattresses You Can Buy OnlineMartin CizmarSecurityUpdate Your iPhone Right Now to Fix 2 Apple Zero DaysDhruv MehrotraBut if we allow vectors to be nearly orthogonal, the number of such distinct vectors in a high-dimensional space explodes. In a 10,000-dimensional space, there are millions of nearly orthogonal vectors.Now let’s create distinct vectors to represent SHAPE, COLOR, CIRCLE, SQUARE, BLUE, and RED. Because there are so many possible nearly orthogonal vectors in a high-dimensional space, you can just assign six random vectors to represent the six items; they’re almost guaranteed to be nearly orthogonal. “The ease of making nearly orthogonal vectors is a major reason for using hyperdimensional representation,” wrote Pentti Kanerva, a researcher at the Redwood Center for Theoretical Neuroscience at the University of California, Berkeley, in an influential 2009 paper.Pentti Kanerva (left) and Bruno Olshausen, researchers at the University of California, Berkeley.Photograph: Chris KymnThe paper built upon work done in the mid-1990s by Kanerva and Tony Plate, at the time a doctoral student with Geoff Hinton at the University of Toronto. The two independently developed the algebra for manipulating hypervectors and hinted at its usefulness for high-dimensional computing.Given our hypervectors for shapes and colors, the system developed by Kanerva and Plate shows us how to manipulate them using certain mathematical operations. Those actions correspond to ways of symbolically manipulating concepts.The first operation is multiplication. This is a way of combining ideas. For example, multiplying the vector SHAPE with the vector CIRCLE binds the two into a representation of the idea “SHAPE is CIRCLE.” This new “bound” vector is nearly orthogonal to both SHAPE and CIRCLE. And the individual components are recoverable—an important feature if you want to extract information from bound vectors. Given a bound vector that represents your Volkswagen, you can unbind and retrieve the vector for its color: PURPLE.The second operation, addition, creates a new vector that represents what’s called a superposition of concepts. For example, you can take two bound vectors, “SHAPE is CIRCLE” and “COLOR is RED,” and add them together to create a vector that represents a circular shape that is red in color. Again, the superposed vector can be decomposed into its constituents.Most PopularScienceThe Seaweed Blob Is Heading to a Beach Near YouChris BaraniukSecurityInside the Dangerous Underground Abortion Pill Market Growing on TelegramLily Hay NewmanGearThe 17 Best (and Worst) Mattresses You Can Buy OnlineMartin CizmarSecurityUpdate Your iPhone Right Now to Fix 2 Apple Zero DaysDhruv MehrotraThe third operation is permutation; it involves rearranging the individual elements of the vectors. For example, if you have a three-dimensional vector with values labeled x, y, and z, permutation might move the value of x to y, y to z, and z to x. “Permutation allows you to build structure,” Kanerva said. “It allows you to deal with sequences, things that happen one after another.” Consider two events, represented by the hypervectors A and B. We can superpose them into one vector, but that would destroy information about the order of events. Combining addition with permutation preserves the order; the events can be retrieved in order by reversing the operations.Together, these three operations proved enough to create a formal algebra of hypervectors that allowed for symbolic reasoning. But many researchers were slow to grasp the potential of hyperdimensional computing, including Olshausen. “It just didn’t sink in,” he said.Harnessing the PowerIn 2015, a student of Olshausen’s named Eric Weiss demonstrated one aspect of hyperdimensional computing’s unique abilities. Weiss figured out how to represent a complex image as a single hyperdimensional vector that contains information about all the objects in the image, including their properties, such as colors, positions, and sizes.“I practically fell out of my chair,” Olshausen said. “All of a sudden, the light bulb went on.”Soon more teams began developing hyperdimensional algorithms to replicate simple tasks that deep neural networks had begun tackling about two decades before, such as classifying images.Consider an annotated data set that consists of images of handwritten digits. An algorithm analyzes the features of each image using some predetermined scheme. It then creates a hypervector for each image. Next, the algorithm adds the hypervectors for all images of zero to create a hypervector for the idea of zero. It then does the same for all digits, creating 10 “class” hypervectors, one for each digit.Now the algorithm is given an unlabeled image. It creates a hypervector for this new image, then compares the hypervector against the stored class hypervectors. This comparison determines the digit that the new image is most similar to.Abbas Rahimi, a computer scientist at IBM Research in Zurich.Courtesy of Abbas RahimiYet this is just the beginning. The strengths of hyperdimensional computing lie in the ability to compose and decompose hypervectors for reasoning. The latest demonstration of this came in March, when Abbas Rahimi and colleagues at IBM Research in Zurich used hyperdimensional computing with neural networks to solve a classic problem in abstract visual reasoning—a significant challenge for typical ANNs, and even some humans. Known as Raven’s progressive matrices, the problem presents images of geometric objects in, say, a 3-by-3 grid. One position in the grid is blank. The subject must choose, from a set of candidate images, the image that best fits the blank.“We said, ‘This is really … the killer example for visual abstract reasoning, let’s jump in,’” Rahimi said.To solve the problem using hyperdimensional computing, the team first created a dictionary of hypervectors to represent the objects in each image; each hypervector in the dictionary represents an object and some combination of its attributes. The team then trained a neural network to examine an image and generate a bipolar hypervector—an element can be +1 or −1—that’s as close as possible to some superposition of hypervectors in the dictionary; the generated hypervector thus contains information about all the objects and their attributes in the image. “You guide the neural network to a meaningful conceptual space,” Rahimi said.Most PopularScienceThe Seaweed Blob Is Heading to a Beach Near YouChris BaraniukSecurityInside the Dangerous Underground Abortion Pill Market Growing on TelegramLily Hay NewmanGearThe 17 Best (and Worst) Mattresses You Can Buy OnlineMartin CizmarSecurityUpdate Your iPhone Right Now to Fix 2 Apple Zero DaysDhruv MehrotraOnce the network has generated hypervectors for each of the context images and for each candidate for the blank slot, another algorithm analyzes the hypervectors to create probability distributions for the number of objects in each image, their size, and other characteristics. These probability distributions, which speak to the likely characteristics of both the context and candidate images, can be transformed into hypervectors, allowing the use of algebra to predict the most likely candidate image to fill the vacant slot.Their approach was nearly 88 percent accurate on one set of problems, whereas neural network-only solutions were less than 61 percent accurate. The team also showed that, for 3-by-3 grids, their system was almost 250 times faster than a traditional method that uses rules of symbolic logic to reason, since that method must search through an enormous rulebook to determine the correct next step.A Promising StartNot only does hyperdimensional computing give us the power to solve problems symbolically, it also addresses some niggling issues of traditional computing. The performance of today’s computers degrades rapidly if errors caused by, say, a random bit flip (a 0 becomes 1 or vice versa) cannot be corrected by built-in error-correcting mechanisms. Moreover, these error-correcting mechanisms can impose a penalty on performance of up to 25 percent, said Xun Jiao, a computer scientist at Villanova University.Hyperdimensional computing tolerates errors better, because even if a hypervector suffers significant numbers of random bit flips, it is still close to the original vector. This implies that any reasoning using these vectors is not meaningfully impacted in the face of errors. Jiao’s team has shown that these systems are at least 10 times more tolerant of hardware faults than traditional ANNs, which themselves are orders of magnitude more resilient than traditional computing architectures. “We can leverage all [that] resilience to design some efficient hardware,” Jiao said.Another advantage of hyperdimensional computing is transparency: The algebra clearly tells you why the system chose the answer it did. The same is not true for traditional neural networks. Olshausen, Rahimi, and others are developing hybrid systems in which neural networks map things in the physical world to hypervectors, and then hyperdimensional algebra takes over. “Things like analogical reasoning just fall in your lap,” Olshausen said. “This is what we should expect of any AI system. We should be able to understand it just like we understand an airplane or a television set.”All of these benefits over traditional computing suggest that hyperdimensional computing is well suited for a new generation of extremely sturdy, low-power hardware. It’s also compatible with “in-memory computing systems,” which perform the computing on the same hardware that stores data (unlike existing von Neumann computers that inefficiently shuttle data between memory and the central processing unit). Some of these new devices can be analog, operating at very low voltages, making them energy-efficient but also prone to random noise. For von Neumann computing, this randomness is “the wall that you can’t go beyond,” Olshausen said. But with hyperdimensional computing, “you can just punch through it.”Despite such advantages, hyperdimensional computing is still in its infancy. “There’s real potential here,” Fermüller said. But she points out that it still needs to be tested against real-world problems and at bigger scales, closer to the size of modern neural networks.“For problems at scale, this needs very efficient hardware,” Rahimi said. “For example, how [do you] efficiently search over 1 billion items?”All of this should come with time, Kanerva said. “There are other secrets high-dimensional spaces hold,” he said. “I see this as the very beginning of time for computing with vectors.”Original story reprinted with permission from Quanta Magazine, an editorially independent publication of the Simons Foundation whose mission is to enhance public understanding of science by covering research developments and trends in mathematics and the physical and life sciences.Get More From WIRED📩 Don’t miss our biggest stories, delivered to your inbox every day🎧 Our new podcast wants you to Have a Nice FutureThe psychedelic scientist who sends brains back to childhoodAbortion pill use is surging. Now it’s under threatHow to live well, love AI, and party like a 6-year-oldThe Reddit blackout is breaking RedditHumans aren’t mentally ready for an AI-filled ‘post-truth world’🌲 Our Gear team has branched out with a new guide to the best sleeping pads and fresh picks for the best coolers and binocularsTopicsQuanta Magazinemathmathematicsartificial intelligenceneural networkscomputer sciencecomputingMore from WIREDThe Seaweed Blob Is Heading to a Beach Near YouThe 5,000-mile-wide Great Atlantic sargassum belt has started to shrink. But it's very difficult to know where its decomposing remains will wash up.Chris BaraniukThe Wild World of Extreme Tourism for BillionairesThe Titan tragedy highlights the burgeoning trend of cavalier high-net-worth individuals exploring some of the most inhospitable places on Earth.Alex ChristianThe Atlantification of the Arctic Ocean Is UnderwayThe discovery of environmental DNA from fish species that have strayed far from their normal range is an ominous reminder of warming, changing seas.William von HerffMissing Sub Passengers Believed Dead After Debris Found From Likely ImplosionWreckage matching the missing submersible was found by a remotely operated vehicle, and so far evidence points to a catastrophic implosion.Aarian MarshallTo Save the Planet, Start DiggingEnvironmentalist Jamie Beard has been pushing her vision to tap into geothermal energy—by working with Big Oil. Can it ever work?Gideon LichfieldBird Populations Are in MeltdownHumans rely on birds to eat insects, spread seeds, and pollinate plants—but these feathered friends can’t survive without their habitats.Chris BaraniukThe Pain and Promise of Europe’s Abortion LawsThe continent’s abortion laws are a patchwork of progress and setbacks. And for many, accessing the right care at the right time is still a lottery.Grace BrowneAbortion Pill Use Is Surging. Now It’s Under ThreatTelehealth allowed patients in many US states to get abortion pills by mail. New legal challenges could block that access—but providers aren't giving up.Emily MullinWIRED is where tomorrow is realized. It is the essential source of information and ideas that make sense of a world in constant transformation. The WIRED conversation illuminates how technology is changing every aspect of our lives—from culture to business, science to design. The breakthroughs and innovations that we uncover lead to new ways of thinking, new connections, and new industries.FacebookTwitterPinterestYouTubeInstagramTiktokMore From WIREDSubscribeNewslettersFAQWired StaffPress CenterCouponsEditorial StandardsBlack FridayArchiveContactAdvertiseContact UsCustomer CareJobsRSSAccessibility HelpCondé Nast StoreDo Not Sell My Personal Info© 2023 Condé Nast. All rights reserved. Use of this site constitutes acceptance of our User Agreement and Privacy Policy and Cookie Statement and Your California Privacy Rights. WIRED may earn a portion of sales from products that are purchased through our site as part of our Affiliate Partnerships with retailers. The material on this site may not be reproduced, distributed, transmitted, cached or otherwise used, except with the prior written permission of Condé Nast. Ad ChoicesSelect international siteUnited StatesLargeChevronUKItaliaJapón", 
            'summary': 'Hyperdimensional computing is an exciting new approach to artificial intelligence that uses hyperdimensional vectors to represent information. This allows machines to reason more abstractly and efficiently, offering advantages over traditional artificial neural networks. With improved transparency and error tolerance, this breakthrough could revolutionize the field of machine learning.', 
            'image': 'https://cdn.stablediffusionapi.com/generations/f3e5e340-220d-4e3b-afd4-f79b6cd3a949-0.png'
        }, 
        {
            'title': 'This AI Scouting Platform Puts Soccer Talent Spotters Everywhere', 
            'url': 'https://www.wired.com/story/ai-football-soccer-scouting/', 
            'text': "This AI Scouting Platform Puts Soccer Talent Spotters Everywhere | WIRED\nSkip to main contentOpen Navigation MenuMenuStory SavedTo revist this article, visit My Profile, then View saved stories.Close AlertThis AI Scouting Platform Puts Soccer Talent Spotters EverywhereBackchannelBusinessCultureGearIdeasScienceSecurityMoreChevronStory SavedTo revist this article, visit My Profile, then View saved stories.Close AlertSign InSearchSearchBackchannelBusinessCultureGearIdeasScienceSecurityPodcastsVideoArtificial IntelligenceClimateGamesNewslettersMagazineEventsWired InsiderJobsCouponsAlex ChristianBusinessJun 1, 2023 7:00 AMThis AI Scouting Platform Puts Soccer Talent Spotters EverywhereA young player hoping to be spotted by a Premier League club typically relies on luck as much as talent, but artificial intelligence could change all that.Photograph: Paul Taylor/Getty ImagesSave this storySaveSave this storySaveIt was the biggest match of Andre Odeku’s career. A week earlier, the 18-year-old forward had been playing in the seventh tier of the English soccer pyramid. Now, he was lining up for Burnley’s Under-23 side in a trial game, with a view to signing for the then-Premier League club’s academy.“Their scouts invited me for a week’s training,” explains Odeku. “I’d never really been on those kinds of pitches before: the grass was so lovely and flat; the ball moved so smoothly and intensely; there were Premier League players training right next to us. They liked what they saw, so they wanted to see me up close in a game.”In footballing terms, Odeku had been plucked from relative obscurity: He’d been discovered while putting in Erling Haaland–like numbers for the development team of semi-professional north London side Haringey Borough; he would go on to score 25 goals in 18 games from the wing and win the league’s golden boot in the 2021–22 season.But Odeku hadn’t been spotted in the traditional way. Weeks before,\xa0in the mud of his local park in East London, he'd docked his phone, hit Record, and begun doing as many push-ups as he could in 30 seconds. As parents and dog walkers strolled by, he sprinted 10 meters, launched into standing jumps, and completed a chest-thumping set of explosive lateral rebound hops.Watching it all was AiSCOUT, a platform that enables footballers to go through virtual trials. Players perform athletic and technical drills on the app, then are rated via an AI scoring system built by data specialists and leading scouts from the game. It’s these talent spotters who decide which few prodigies will fulfill their childhood dream of becoming a professional footballer and which won’t end up having a football career at all.The founder of AiSCOUT, Darren Peries, began developing the app following his son’s release by Spurs, aged 16. “Scouts from elsewhere had nothing on him in terms of information, game footage, or training metrics,” explains Richard Felton-Thomas, the platform’s COO and director of sport science. “While there is much data collection in the senior professional game, there isn’t the same infrastructure in youth football—even at the elite level.”For KicksThe World Cup, Netflix, and the Sanitized Future of SportsAmit KatwalaDigital Culture NFTs Are Conquering SoccerAmit KatwalaDouble DownHas the US Learned Nothing From the UK’s Gambling Woes?Will BedingfieldFollowing seven months of live testing, and the\xa0analysis of millions of data points, AiSCOUT machine learning is now able to\xa0measure players’ biomechanics, technique, and athletic prowess down to the minutiae; feedback is automated and delivered via the app within the hour. After players perform core athletic drills, the best are invited to show off their on-the-ball skills: Odeku’s virtual Burnley trial included an agility dribble and seven-cone weave in the park; his blistering speed and ball control earned the teenager an invite to the club’s Barnfield training center in Lancashire.In a game that’s often ruthless in its discarding of young footballers, AiSCOUT offers a second chance—to both players and scouts. Odeku was released from Arsenal aged 11, then Brentford when he was 13; his diminutive size was cited as a factor. “Academies have a ‘win now’ mentality, so smaller players yet to mature are often let go,” says Felton-Thomas. “Once that happens, there are no eyes on them, even if they’ve had a growth spurt. Now, players can be back in the system via AiSCOUT, with clubs able to keep tabs on their progress.”Most PopularScienceThe Seaweed Blob Is Heading to a Beach Near YouChris BaraniukSecurityInside the Dangerous Underground Abortion Pill Market Growing on TelegramLily Hay NewmanGearThe 17 Best (and Worst) Mattresses You Can Buy OnlineMartin CizmarScienceThe Wild World of Extreme Tourism for BillionairesAlex ChristianSpotting a future Premier League superstar typically involves club scouts spending long hours on the road, traveling to half-empty community grounds and mud-caked Sunday League pitches. Relying upon an expert eye and word-of-mouth, there is more hope than expectation that they’re about to unearth the next footballing gem.Felton-Thomas says the data-driven platform, founded in 2018 and now an official academy partner of both Burnley and Premier League giant Chelsea, could become an invaluable tool in youth player recruitment. “We help identify talent the scouts never knew existed. There are too many great young footballers slipping through the system. The technology helps target the right players faster, with data and insights to make better decisions and recruit down to Under-8 level based on whatever metrics a club is prioritizing.”Odeku got an assist in his trial game with Burnley. He’s subsequently been invited back for another match. From doing sprints and jumps in his local park to landing a Premier League academy trial, he says AiSCOUT has given him the gift that every young, promising footballer craves: confidence. “I came away believing I could make it,” he says.The technology has also offered Odeku and many other young players an opportunity. “Normally, it’s based on luck whether you’re scouted: You’ve got to have consistently unreal performances and stats, hope the right people hear about you, then wait for them to watch you play live—if you even have a good game,” says Odeku, who is now in Haringey’s senior side. “Now, with this technology, it’s facts- and stats-based: It’s another avenue for young players to be seen and have a chance of making it in professional football.”This article first appeared in the January/February 2023 edition of WIRED UK.Get More From WIRED📨 Understand AI advances with our Fast Forward newsletter🎧 Our new podcast wants you to Have a Nice FutureHow Christopher Nolan learned to stop worrying and love AIA Chinese firm’s encryption chips got inside the Navy and NASAPodcasts could unleash a new age of enlightenmentApple is taking on apples in a weird trademark battleFighting games are a haven for LGBTQ+ gamers🔌 Charge right into summer with the best travel adapters, power banks, and USB hubsTopicssoccerSportsartificial intelligenceMore from WIREDThey Plugged GPT-4 Into Minecraft—and Unearthed New Potential for AIThe bot plays the video game by tapping the text generator to pick up new skills, suggesting that the tech behind ChatGPT could automate many workplace tasks.Will KnightMicrosoft’s Satya Nadella Is Betting Everything on AIThe CEO can’t imagine life without artificial intelligence—even if it’s the last thing invented by humankind.\xa0Steven LevyMeet the AI Protest Group Campaigning Against Human ExtinctionFears that artificial intelligence might wipe us out have fueled the rise of groups like Pause AI. Their warnings aren’t that far-fetched, experts say.Morgan MeakerMillionaires Are Begging Governments to Tax Them MoreA group of multimillionaires say it's time to get serious about taxing wealth. They argue that this would promote economic stability and benefit everyone.Sophie ChararaRunaway AI Is an Extinction Risk, Experts WarnA new statement from industry leaders cautions that artificial intelligence poses a threat to humanity on par with nuclear war or a pandemic.Will KnightAll the Ways ChatGPT Can Help You Land a JobWhether you use ChatGPT, Bard, or Bing, your favorite AI chatbots can help your application stand out from the crowd.David NieldHow\xa0AI Protects (and Attacks) Your InboxCriminals may use artificial intelligence to scam you. Companies, like Google, are looking for ways AI and machine learning can help prevent phishing.Reece RogersWhy the Story of an AI Drone Trying to Kill Its Operator Seems So TrueA widely shared—and false—story highlights the need for greater transparency in the development and engineering of AI systems.Will KnightWIRED is where tomorrow is realized. It is the essential source of information and ideas that make sense of a world in constant transformation. The WIRED conversation illuminates how technology is changing every aspect of our lives—from culture to business, science to design. The breakthroughs and innovations that we uncover lead to new ways of thinking, new connections, and new industries.FacebookTwitterPinterestYouTubeInstagramTiktokMore From WIREDSubscribeNewslettersFAQWired StaffPress CenterCouponsEditorial StandardsBlack FridayArchiveContactAdvertiseContact UsCustomer CareJobsRSSAccessibility HelpCondé Nast StoreDo Not Sell My Personal Info© 2023 Condé Nast. All rights reserved. Use of this site constitutes acceptance of our User Agreement and Privacy Policy and Cookie Statement and Your California Privacy Rights. WIRED may earn a portion of sales from products that are purchased through our site as part of our Affiliate Partnerships with retailers. The material on this site may not be reproduced, distributed, transmitted, cached or otherwise used, except with the prior written permission of Condé Nast. Ad ChoicesSelect international siteUnited StatesLargeChevronUKItaliaJapón", 
            'summary': 'AiSCOUT is an AI scouting platform that is changing the game for young soccer players. By using virtual trials and AI algorithms, talented players who may have been overlooked by traditional scouting methods can now showcase their skills and get discovered. This innovative platform has already partnered with Premier League clubs and has the potential to revolutionize youth player recruitment.', 
            'image': 'https://cdn.stablediffusionapi.com/generations/c6860025-c48d-467f-b344-b2c0fe9e3ce4-0.png'
        }
    ], 
    'abstract': 'Hyperdimensional computing is revolutionizing Generative AI by using hyperdimensional vectors to represent information, leading to more abstract and efficient reasoning. Additionally, AiSCOUT, an AI scouting platform, is changing the game for young soccer players by using virtual trials and AI algorithms to help talented players get discovered and revolutionize youth player recruitment.', 
    'title': 'Revolutionizing Generative AI: Hyperdimensional Computing and AI Scouting Platforms Take Center Stage'
}