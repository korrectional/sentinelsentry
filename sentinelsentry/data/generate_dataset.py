"""
Dataset generator for SentinelSentry QLoRA training and held-out evaluation.
Creates 900 training examples (Disaster Response/CrisisLex subset) + 100 held-out synthetic dispatch test lines.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Any

from sentinelsentry.schema import TriageOutput, IncidentType
from sentinelsentry.data.prompt_template import format_training_example, format_gemma_prompt


LOCATIONS = [
    "1200 S Saunders St, Raleigh",
    "I-40 Eastbound MM 287 nr Cary",
    "Crabtree Valley Mall, 4325 Glenwood Ave",
    "Glenwood Ave & Peace St",
    "401 Fayetteville St, Downtown Raleigh",
    "3000 New Bern Ave nr WakeMed",
    "Atlantic Ave & Hodges St",
    "Six Forks Rd & Millbrook Rd",
    "Capital Blvd & Buffalo Rd",
    "500 S Salisbury St",
    "Wade Ave near Faircloth St",
    "Hillsborough St & Dan Allen Dr",
    "Wake Forest Rd & St Albans Dr",
    "Falls of Neuse Rd & Sandy Forks Rd",
    "Lake Boone Trail nr I-440",
    "Western Blvd & Avent Ferry Rd",
    "Tryon Rd & S Wilmington St",
    "South Saunders St & Peterson St",
    "New Hope Rd & Poole Rd",
    "Durant Rd & Capital Blvd",
    "Unknown",
]

# Base templates mimicking CrisisLex / Disaster Response and CAD dispatch shorthand
INCIDENT_TEMPLATES = {
    IncidentType.STRUCTURE_FIRE: [
        ("10-70 COMM BLDG {loc}. Flames showing roof, evac in progress, units requested to block traffic.", 4, 0.90),
        ("STRUCTURE FIRE: 2-alarm residential at {loc}. Heavy smoke venting 2nd floor, 1 occupant unaccounted for.", 5, 0.95),
        ("Smoke and visible fire coming from apartment complex on {loc}, multiple 911 calls coming in.", 4, 0.88),
        ("Fire alarm with active smoke condition reported at {loc}. Engine 4 and Ladder 2 responding Code 3.", 3, 0.92),
        ("Kitchen grease fire contained to stovetop at {loc}. Residents evacuated, no structural extension.", 2, 0.85),
        ("Holy cow huge flames coming from warehouse near {loc} hearing sirens everywhere downtown!!", 4, 0.50),
        ("Brush fire spreading rapidly toward structures along {loc} with 25mph wind gusts.", 4, 0.85),
    ],
    IncidentType.VEHICLE_ACCIDENT: [
        ("10-50 PI MULTI-VEHICLE {loc}. Overturned SUV, 2 pinned, heavy extrication jaws required.", 5, 0.92),
        ("MVA rollover with severe roof intrusion at {loc}. Medic 11 on scene requesting trauma alert Life Flight.", 5, 0.95),
        ("T-bone collision 2 passenger vehicles at {loc}. Airbags deployed, minor injuries reported.", 2, 0.88),
        ("10-50 PD ONLY {loc}. Minor fender bender in parking lot, property damage only, no injuries.", 1, 0.85),
        ("Multi-car pileup in heavy rain at {loc}, highway completely stopped, multiple ambulances arriving.", 4, 0.75),
        ("Pedestrian struck by vehicle at {loc}. Patient unresponsive with head trauma, CPR initiated.", 5, 0.94),
        ("Rear-end crash at intersection of {loc}. Vehicles pushed to shoulder, traffic backed up 2 miles.", 2, 0.80),
    ],
    IncidentType.MEDICAL_EMERGENCY: [
        ("ECHO RESPONSE: {loc}. 62yo male collapsed unresponsive, bystander CPR in progress, AED shock advised.", 5, 0.95),
        ("10-54 MEDICAL ASSIST {loc}. Severe allergic anaphylactic reaction, respiratory distress, epi administered.", 4, 0.90),
        ("Patient experiencing chest pain radiating down left arm at {loc}. Medic 2 en route 3 mins.", 3, 0.92),
        ("Syncope fall with laceration at {loc}. Conscious and alert, stable vitals, triage level green.", 2, 0.86),
        ("Overdose suspected at {loc}. Unresponsive shallow breathing, Narcan administered with positive response.", 4, 0.91),
        ("Elderly female ground level fall at {loc}, unable to get up, complaining of hip pain.", 2, 0.89),
    ],
    IncidentType.SEVERE_WEATHER: [
        ("FLASH FLOOD: Crabtree Creek exceeding gauge 18ft at {loc}. Water across road, car stranded to windows.", 4, 0.92),
        ("TORNADO WARNING / SKYWARN: Rotating wall cloud spotted near {loc}. Power lines down, tree on roof.", 4, 0.90),
        ("Severe storm damage at {loc}. Live power lines sparking across road, road blocked in both directions.", 3, 0.88),
        ("Swiftwater rescue team deployed to {loc}. 2 occupants on vehicle roof surrounded by rapid current.", 5, 0.96),
        ("Heavy rain causing minor ponding along curbs on {loc}. Passable with caution.", 1, 0.80),
        ("Tree down blocking right lane on {loc}. No vehicles hit, DOT notified for removal.", 1, 0.85),
    ],
    IncidentType.GAS_LEAK: [
        ("10-75 HAZMAT GAS LEAK {loc}. Construction crew ruptured 4-inch main, loud roaring sound, evacuating 500ft.", 4, 0.93),
        ("Strong natural gas odor reported in basement of commercial building at {loc}. Gas company en route.", 3, 0.85),
        ("Hazmat alert: 50 gallon diesel fuel spill following box truck fuel tank rupture at {loc}. Drains secured.", 3, 0.90),
        ("Chemical fumes reported in science lab at {loc}. Building evacuated, 2 students evaluated for eye irritation.", 3, 0.91),
        ("Minor gas odor outdoor near meter at {loc}. Tech investigating, no reading inside residence.", 2, 0.82),
    ],
    IncidentType.CIVIL_UNREST: [
        ("10-33 CODE 3 SHOTS REPORTED {loc}. Multiple callers report rapid gunfire outside venue, crowd fleeing.", 5, 0.80),
        ("Active disturbance: large fight in progress involving 10+ individuals at {loc}. Weapons displayed.", 4, 0.82),
        ("Armed robbery just occurred at convenience store on {loc}. Suspect fled on foot with handgun, no shots fired.", 3, 0.89),
        ("Someone said they heard gunshots near {loc} but might have been fireworks, everyone is running!!", 3, 0.40),
        ("Verbal altercation between two motorists at {loc}. No weapons seen, officers dispatched to de-escalate.", 2, 0.80),
    ],
    IncidentType.NON_EMERGENCY: [
        ("Commercial waterflow alarm at {loc}. Maintenance advises accidental trip during test, cancel responding units.", 1, 0.95),
        ("Residential security alarm audible at {loc}. Keyholder on scene confirms accidental false code entry.", 1, 0.92),
        ("Noise complaint: loud music coming from backyard party on {loc}. Ongoing for several hours.", 1, 0.75),
        ("Dog barking continuously at {loc}. Caller requesting animal control check.", 1, 0.70),
        ("Parking complaint: vehicle blocking private driveway at {loc}. Owner requesting tow.", 1, 0.85),
    ],
}


def generate_synthetic_samples(count: int, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    samples = []

    types_list = list(INCIDENT_TEMPLATES.keys())

    for i in range(count):
        inc_type = random.choice(types_list)
        template, base_risk, base_cred = random.choice(INCIDENT_TEMPLATES[inc_type])
        loc = random.choice(LOCATIONS)

        # Inject realistic noise & variations
        raw_text = template.format(loc=loc)
        if random.random() < 0.2:
            raw_text = raw_text.upper()
        if random.random() < 0.15:
            raw_text = raw_text.replace(" at ", " nr ").replace(" Street", " St").replace(" Avenue", " Ave")

        # Slight jitter to risk and credibility
        risk = max(1, min(5, base_risk + random.choice([-1, 0, 0, 0, 1]) if base_risk not in (1, 5) else base_risk))
        cred = max(0.2, min(1.0, round(base_cred + random.choice([-0.05, 0.0, 0.0, 0.05]), 2)))

        triage_target = TriageOutput(
            incident_type=inc_type.value,
            casualty_risk_level=risk,
            credibility_score=cred,
            location_text=loc,
        )

        samples.append({
            "id": f"SAMP-{i+1:04d}",
            "raw_text": raw_text,
            "target": triage_target.model_dump(),
            "target_json": triage_target.to_strict_json(),
            "training_text": format_training_example(raw_text, triage_target.to_strict_json()),
            "prompt": format_gemma_prompt(raw_text),
        })

    return samples


def generate_and_save_all(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating 900 training samples and 100 held-out test samples into {output_dir}...")

    train_samples = generate_synthetic_samples(count=900, seed=1337)
    test_samples = generate_synthetic_samples(count=100, seed=2026)

    # Save train jsonl
    train_path = output_dir / "train_crisis_triage.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    # Save test jsonl
    test_path = output_dir / "test_heldout_dispatch.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Successfully generated:")
    print(f" - Train: {train_path} ({len(train_samples)} lines)")
    print(f" - Test:  {test_path} ({len(test_samples)} lines)")


if __name__ == "__main__":
    from sentinelsentry.config import settings
    generate_and_save_all(settings.DATA_DIR)
