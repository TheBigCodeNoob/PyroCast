# PyroCast

This is a model that tries to predict where a wildfire will start in the southeastern US, a day or two before it happens. Not how a fire spreads once it's going — where a brand new one will pop up.

I should say up front that the accuracy number isn't really the point of this project, even though I'm happy with where it landed. The actual story is that I kept building models that looked amazing and then realizing they were cheating. Most of the time I spent on this was figuring out *how* each one was cheating and taking the cheat away. The number only means something because of all that, so that's most of what this writeup is about.

If you just want the bottom line: the final model scores **0.852 AUC** on places and years it was never trained on. I'm fairly confident that's real (95% confidence interval 0.844–0.860), and even if you're maximally suspicious and strip out every "easy" clue, there's still real skill left underneath (~0.79). More on what all of that means below.

There are two halves to this file. The first half is the detailed version. The second half ("Explain it like I'm not a nerd") is the same thing in plain language for anyone who doesn't do machine learning — including judges. They cover the same ground, so read whichever one fits you.

---

## The detailed version

### What it actually does

You give it a spot on the map and a date. It gives back a probability that a wildfire ignites at that spot within the next couple of days. I freeze all the input data at **two days before** the fire was actually discovered, so the model is genuinely forecasting and never gets to peek at the fire's own conditions.

A few things it is deliberately *not*, because people mix these up:

- It's not a fire-*spread* model. Predicting how an existing fire grows is a different (and pretty well-studied) problem.
- It's not a static "risk map." Those have no time dimension and they tend to just memorize which regions are fire-prone. My very first model did exactly that, and it fooled me.
- It's not a big-fire model. A lot of fire datasets only record large fires, and large fires happen in remote wilderness, so "predicting large fires" quietly becomes "predicting remoteness." That also fooled me.

The study area is the southeastern US — eight states, roughly latitude 24.5 to 37.5 and longitude −94.5 to −75. The project folder is named "Florida" for historical reasons (that's where I started), but Florida by itself turned out to be too flat and uniform to learn much from, so it grew to the whole Southeast. I'll come back to that, because it's an honest weakness.

### The part that actually matters: how I test it

Here's the trap with this kind of model. It can score a gorgeous accuracy number and be completely useless, because the test is secretly leaking the answer. Almost all of my effort went into not falling for that. Three specific leaks bit me:

1. **Geography leak.** The model learns "this whole *region* burns a lot" instead of "a fire starts *here, today*." You catch this by testing only on areas the model never saw during training.
2. **Season leak.** If your real fires are mostly summer and your "no fire" examples are random months, the model just learns "summer = fire." Everybody already knows summer is dangerous; that's not a useful prediction. You catch this by making the no-fire examples come from the same months as the real fires.
3. **Collection-bias leak.** Sometimes the positives and negatives differ because of *how the data was gathered*, not because of fire. Big fires only get recorded in remote places. Fires near towns get reported more often. You catch this by forcing the fire and no-fire examples to have the same distribution of the suspicious variable, then checking whether the score holds up.

So the number I trust — the one quoted everywhere in this file — comes from what I call the blocked space-and-time test. I chop the map into 1-degree grid cells and make sure a whole cell is either training or testing, never split. Then I train only on 2017–2019 and test only on 2020. The model has to get it right for **new places in a future year, at the same time.** On top of that, every improvement gets re-checked with the matching trick from leak #3 before I believe it.

That rule — throw out any gain that doesn't survive the leak checks — is why my accuracy went *down* before it honestly went up. I'll get to that.

### Everything that went wrong (the actual story)

This is in order. Each version either exposed a cheat or earned a real gain.

**v2 — the 0.95 that wasn't real.** My first real model was a neural network on satellite imagery, and it scored 0.95. I was thrilled for about a day. Then I tested it on regions it hadn't trained on and it fell apart. It had learned which *ecoregions* tend to burn — basically memorizing the map — not anything about when or where a fire would actually start. This was the most important thing I learned in the whole project: a big number with no spatial holdout is meaningless.

**v3 — the first honest one.** I rebuilt everything with proper holdouts and matched no-fire examples. The honest scores were 0.69 for the neural net and 0.73 for a plain logistic regression. Way less impressive, but at least now I was measuring reality instead of my own wishful thinking.

**v6 — 0.93, and it was the calendar.** I switched to a different fire dataset and got 0.93. By this point I was suspicious of any number that good, and sure enough, my no-fire examples weren't season-matched. The model had learned "fire season." I fixed the season matching, which led to v7.

**v7 — 0.80, and it was remoteness.** Season-matched, this dataset scored 0.80. But this dataset (MTBS) only contains *large* fires, and large fires happen out in the wilderness. When I forced the fire and no-fire points to have the same population density around them, the 0.80 dropped to about 0.67. So the 0.80 was mostly "big fires happen where nobody lives," which is true but useless. (Around the same time I checked individual features and found temperature and humidity also looked like strong predictors until I season-matched, at which point they collapsed too.)

**v8 — the honest baseline, 0.716.** I moved to satellite active-fire detections, which catch real fire occurrence — small fires, fires near people — not just the big remote ones. The score was 0.716 with population matching, which is *lower* than v7's 0.80 but actually honest (matching barely changed it, which is the sign there's no remoteness cheat left). I locked this in as my baseline. The one annoying thing about this dataset is that the Southeast does a lot of prescribed (intentional) burning, which is a different thing from wildfire.

**v9 — feature engineering hit a wall.** I threw every weather and drought feature I could think of at v8 — burning index, fuel moisture, wind, dry-day streaks, longer drought windows. Total improvement: 0.001. Basically nothing. The weather signal was already saturated. That told me the ceiling wasn't my cleverness with features, it was the data itself.

**v10b — the real breakthrough.** I found FPA-FOD, a federal database of *actual* wildfire ignitions with the cause attached, and it excludes prescribed burns. About 94% of these fires are started by people. So "how close is this to where people live and work" became a genuinely useful clue — and importantly, I could show it's a *real* cause, not a data artifact. Human-caused fires sit a median of about 30 meters from development, lightning fires (which people don't start) sit about 70 meters out, and random background points about 90 meters out. The *cause* determines *where*, which is exactly what you'd see if people actually do start fires near themselves. This got me to an honest 0.811, my first real gain over the 0.716 baseline. One design detail that mattered: I drew the no-fire points from *all* land, including developed areas, not just wildland — otherwise the model could've won by just learning "developed = fire."

**v11 — grinding from 0.81 to 0.85.** This was a long string of small, carefully-checked improvements. Each one I added, re-checked against the leak tests, and either kept or threw out:

- Looking at the *landscape around* each point — how much forest, wetland, and farmland is nearby. Helped.
- Farmland context specifically (crop and pasture fraction). A surprising amount of Southern fire is agricultural and debris burning, so this helped more than I expected.
- Feeding it more data — 10,000 fires up to 25,000. This was also a test of whether I'd been fooling myself: the 15,000 new fires were ones the model had never been tuned on, so if my gains were fake they'd disappear here. They didn't. They held and even improved slightly, which put my biggest fear (that I was overfitting the test) to rest.
- Terrain — slope, ruggedness, which way the hill faces. **Nothing.** The Southeast is flat. Terrain matters for fire out West, not here. A null result, but a real finding.
- The last real winner: how *tall* the vegetation is. A satellite measures tree-canopy height and tree cover, and it turns out tall dense pine is very different fuel than low scrub or open marsh — and that vertical structure was a clue the model had been missing. This is the gain I trust the most, because plant height literally can't be a "reporting bias." It's just measuring plants. This pushed it over 0.85.

If you want the one-line summary of the climb: 0.81 → 0.827 → 0.835 → 0.840 → 0.852, and every single step survived the leak checks.

### Okay, but how do I know it's not cheating *again*?

Fair question, since I'd been wrong four times by this point. So I ran a pile of "where could I still be fooling myself" tests:

- **Shuffle test.** I randomly scrambled the fire / no-fire labels and re-ran the whole pipeline. It scored 0.515 — a coin flip. If my code were leaking the answer somewhere, scrambled labels would still score high. They didn't, so the machinery is clean.
- **Garbage feature test.** I added a column of pure random numbers. The model ignored it. It's not just grabbing onto anything.
- **Is the cheat hiding somewhere I didn't check?** I matched the fire and no-fire points on each suspicious variable one at a time. Matching on elevation, greenness, crop fraction — none of those changed the score, so the cheat isn't hiding there. The only things that move the score when you match on them are the human-access features, which I already knew about and already argued are mostly a real cause.

The honest result is a range, not one triumphant number, depending on how strict you want to be:

| How skeptical you're being | Score |
|---|---|
| Normal — people really do cause 94% of these fires | 0.852 |
| Remove the "remoteness" advantage (population-matched) | 0.811 |
| Strictest single correction | 0.801 |
| Delete *every* human-related clue, leave only nature | ~0.79 |

Even with every human clue deleted, there's about 0.79 of genuine skill from weather, drought, and fuel/canopy. So I'm comfortable saying the real answer lives somewhere in 0.79–0.85, and I'd rather show the whole range than pretend it's a clean 0.852 with no asterisk.

### Where it ended up (the numbers)

The final model is `best_model_v11h.joblib`. Tested on new places in a future year:

| What I measured | Score |
|---|---|
| Headline AUC | 0.852 (95% CI 0.844–0.860) |
| Population-matched (no remoteness advantage) | 0.811 |
| Strictest single correction | 0.801 |
| Nature-only floor (all human clues removed) | ~0.79 |
| Hold out a whole sub-region, predict it cold (8 regions) | 0.841 average, 0.794 worst |
| Florida only | 0.751 |
| Calibration (Brier score, lower is better) | 0.176 vs 0.241 baseline |

### What it's bad at

I'd rather you hear this from me than find it yourself.

- **Florida specifically is weaker, around 0.75.** The 0.852 leans on the *variety* of the whole Southeast — different forests, land uses, canopy. Flat uniform Florida gives the model less to work with, and the canopy improvement that pushed me over 0.85 added basically nothing in Florida. The project is named for Florida and it's worst in Florida, which is a little embarrassing but true.
- **It's much better at "where" than "when."** Picking the *spot* is strong. Picking the exact *day* is weak (~0.64–0.73), because a person deciding to burn yard waste on a random Tuesday is close to unpredictable. I ran a separate experiment that confirmed the timing signal does exist and is learnable, but it's a different axis from the main score. A natural next step would be a combined where-and-when map.
- **0.852 is a ranking score, not "85% of fires caught."** Real fires are rare, so at the true rate you'd still get false alarms. What the model is genuinely good at is *prioritizing* — telling you where to look first — which is what's actually useful.

### Technical reference

**Data.** Positives are real wildfire ignitions from FPA-FOD (52,633 in the SE, 2017–2020, ~94% human / ~5% lightning, prescribed burns excluded), pulled from the US Forest Service ArcGIS server. Negatives are random non-water land points, dated to match the fires' month distribution. All features frozen at discovery minus 2 days. Final dataset is 25,000 positives plus matched negatives.

**Features (~52).** Weather from GRIDMET (precip over 7–365 days, vapor-pressure deficit, energy release component, 100-hour fuel moisture, max temp, min humidity); drought (PDSI at several lags); greenness (MODIS NDVI/EVI); human access (population, distance-to-developed, developed fraction, VIIRS nighttime lights); NLCD land-cover fractions; neighborhood land-cover fractions in 0.5–5 km rings; canopy height (ETH 10 m) and tree cover (Hansen); elevation; plus a few engineered drought/dryness trends. All computed in Google Earth Engine.

**Model.** Gradient-boosted trees (scikit-learn's `HistGradientBoostingClassifier`). I checked it isn't model-specific — logistic regression gets ~0.78, random forest ~0.81, LightGBM ~0.81, so the result holds across model types and even a linear model does fine. Early on I tried a ResNet CNN on image stacks; the trees on tabular features matched it and are far easier to interpret.

**Reproducing it.**

```
python fpafod_extract.py        # pull real ignitions -> fpafod_se.csv
python Dataget_v11e.py          # export 25k positives + features from Earth Engine
python Dataget_v11h_canopy.py   # add canopy height + tree cover
                                # monitor_*.py download the exports as they finish
python model_v11h.py            # evaluate (blocked space+time + leak checks)
python save_best_v11h.py        # train + save best_model_v11h.joblib
python crutch_hunt.py           # the honesty audit
```

`EXPERIMENTS.md` is my actual lab notebook — every experiment with its numbers, in the order I ran them.

---

## Explain it like I'm not a nerd

Here's the whole thing again with no jargon. If you're a judge, this is the part to read.

**What I built.** A program that looks at a spot on a map and a day, and says "a wildfire is likely to start here" a day or two before it happens. Like a weather forecast, except instead of rain it's predicting where a *new* fire will break out across the southeastern US.

**Why this is way harder than it sounds.** You'd think you just show a computer a bunch of old fires and let it figure out the pattern. The problem is the computer will happily learn the *wrong* pattern and look like a genius doing it.

Picture this. You want to predict which students get an A. You build a model and it's 95% accurate — incredible. Then you look closer and realize it's just checking which *school* each kid goes to. It never learned anything about the actual students; it memorized "fancy school = A." Hand it a new kid at a new school and it's useless. That's the exact trap a fire model falls into: it memorizes "this *region* burns" instead of learning "a fire will start *here, today*." My whole project was a fight against that.

**I watched my own model cheat four times.** Not exaggerating:

1. My very first model scored 95% and I was pumped. Turned out it just memorized which *types of land* burn — the "fancy school" trick. When I made it predict for places it had never seen, the 95% vanished.
2. A later one scored 93% by secretly learning "summer = fire." True, but useless — everyone knows summer is dangerous. I fixed it by making the no-fire examples come from the same months as the real fires.
3. Another scored 80% by learning "the middle of nowhere = fire," because that dataset only had big fires and big fires happen in the wilderness. Correcting for that knocked it down to 67%.
4. Each time, I deleted the cheat and accepted the lower honest score. My project's number went *down* — from a fake 95% to an honest 69% — before I earned it back the right way.

**Then I built it back up honestly.** I found a much better record of real fires, one that tells you the cause. About 94% of these fires are started by people, so "how close is this to where people live" became a real, useful clue. And I could prove it's real and not another trick: people's fires happen right next to towns (about 30 meters out), lightning fires — which people don't start — happen much farther out (about 70 meters), and random spots farther still. The cause decides the location. That's a genuine pattern, not an accident.

From there I improved it one careful step at a time: looking at what's *around* each spot (nearby forest, wetland, farmland), realizing a ton of Southern fires come from farm and debris burning, feeding it way more data, and finally — the big one — using satellites to measure how *tall* the plants are. Tall dense pine is totally different fuel than short scrub or marsh, and that was the missing piece. That's the improvement I trust the most, because the height of a tree literally cannot be a "reporting" trick. It's just measuring the tree. That's what pushed me over 85%.

**What "85%" actually means.** If I hand the model one real fire location and one random spot, 85% of the time it correctly says the fire spot is riskier. And practically, if you took the 5% of places it flags as most dangerous on a given day, the real fires are heavily concentrated in that slice. I'm honest that it's a range depending on how strict you are — 85% if you accept "fires start near people" as a real fact (it is), down to about 79% if you paranoidly delete every single human-related clue and leave only nature. Even then, there's real skill there.

**How do I know I'm not fooling myself a fifth time?** This is the part you should grill me on, so here's my answer ahead of time. I scrambled the fire / no-fire labels randomly and re-ran everything — it dropped to a coin flip (51.5%). If my code were secretly leaking the answer, scrambled labels would still score high; they didn't, so the machinery is clean. I also added a column of pure random noise and the model ignored it. And I never, ever test it on a place or year it trained on.

**What it's bad at, because I'd rather say it than hide it.** It's weaker in Florida specifically (~75%) — ironic, since the project's named for Florida — because Florida is flat and same-y and there's less for the model to grab onto; the 85% comes from the variety of the whole region. It's much better at guessing *where* than *what day*, because a person deciding to burn yard waste on a Tuesday is basically random. And "85%" is a ranking score, not "85% of fires caught" — in the real world fires are rare, so you'd still get some false alarms. What it's genuinely good at is telling you where to look first.

**Why I think this matters.** Wildfire ignition in the Southeast is weirdly understudied — most fire research is about the western US or about how fires spread, not where southeastern fires start. But honestly, the thing I'm proudest of isn't the 85%. It's that I caught myself cheating four separate times and fixed it each time, instead of reporting the flashy 95% and calling it a day. The number is something I can actually defend, and I can show my work for every piece of it.
