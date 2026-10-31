"""
Realistic dispatch items, radio shorthands, and cross-feed corroboration scenarios.
"""

from typing import Dict, List, Any

# Pre-scripted realistic dispatch and crisis stream events
CRISIS_DISPATCH_SCENARIOS: List[Dict[str, Any]] = [
    # ------------------ High-Impact Scenario 1: Warehouse 2-Alarm Fire ------------------
    {
        "scenario_id": "SCN-001-WAREHOUSE-FIRE",
        "description": "2-Alarm commercial warehouse fire with collapse risk and trapped worker",
        "items": [
            {
                "feed_id": "cad_police",
                "title": "CAD 911: 10-70 COMM BLDG 1200 S SAUNDERS ST",
                "raw_text": "10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF EVAC IN PROGRESS UNITS REQ TO BLOCK SAUNDERS FROM SOUTH TO PETERSON. 1 SUBJ REPORTED INSIDE.",
                "ground_truth": {
                    "incident_type": "Structure Fire",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.90,
                    "location_text": "1200 S Saunders St, Raleigh",
                },
            },
            {
                "feed_id": "fire_rescue",
                "title": "WAKE FIRE DISPATCH: 2-ALARM COMM FIRE",
                "raw_text": "STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St, Raleigh. Engine 3 Truck 1 on scene, heavy black smoke venting through roof, collapse zone established, 1 occupant unaccounted for.",
                "ground_truth": {
                    "incident_type": "Structure Fire",
                    "casualty_risk_level": 5,
                    "credibility_score": 0.95,
                    "location_text": "1200 S Saunders St, Raleigh",
                },
            },
            {
                "feed_id": "social_crisis",
                "title": "Citizen X: Huge explosions warehouse Saunders St",
                "raw_text": "Holy smoke huge explosions coming from warehouse district on Saunders St near downtown Raleigh right now sirens everywhere!! Thick black smoke filling sky",
                "ground_truth": {
                    "incident_type": "Structure Fire",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.50,
                    "location_text": "Saunders St near downtown Raleigh",
                },
            },
            {
                "feed_id": "ems_scanner",
                "title": "EMS TAC-2: STANDBY 1200 S SAUNDERS ST",
                "raw_text": "EMS 11 DISPATCH: Standby 1200 S Saunders St commercial 2nd alarm. Medic 4 staging on Commerce Pkwy for civilian burn victim extraction.",
                "ground_truth": {
                    "incident_type": "Structure Fire",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.92,
                    "location_text": "1200 S Saunders St",
                },
            },
        ],
    },

    # ------------------ High-Impact Scenario 2: Multi-Vehicle Rollover with Extrication ------------------
    {
        "scenario_id": "SCN-002-MVA-EXTRICATION",
        "description": "Multi-vehicle rollover on I-40 with critical entrapment and highway shutdown",
        "items": [
            {
                "feed_id": "cad_police",
                "title": "CAD 911: 10-50 PI I-40 EB MM 287 NR CARY",
                "raw_text": "10-50 PI MULTI-VEHICLE I-40 EB MM 287 NR CARY EXT RAMP. OVERTURNED SUV 2 ADULTS 1 PED TRAPPED LANES BLOCKED SHUT DOWN ALL EASTBOUND LANES CODE 3.",
                "ground_truth": {
                    "incident_type": "Vehicle Accident / Rescue",
                    "casualty_risk_level": 5,
                    "credibility_score": 0.92,
                    "location_text": "I-40 Eastbound MM 287 near Cary",
                },
            },
            {
                "feed_id": "fire_rescue",
                "title": "FIRE OPS: HEAVY EXTRICATION I-40 MM 287",
                "raw_text": "RESCUE SQUAD 1 DISPATCHED: I-40 EB Mile Marker 287 heavy extrication required, rollover with severe roof intrusion, 2 critical trapped jaws in service.",
                "ground_truth": {
                    "incident_type": "Vehicle Accident / Rescue",
                    "casualty_risk_level": 5,
                    "credibility_score": 0.95,
                    "location_text": "I-40 EB Mile Marker 287",
                },
            },
            {
                "feed_id": "ems_scanner",
                "title": "EMS TRAUMA ALERT: 3 PATIENTS I-40 CARY",
                "raw_text": "TRAUMA CODE: 3 patients I-40 EB Mile 287 Cary exit. 1 pediatric red tag critical, 2 yellow tag trauma, Duke Life Flight requested on standby.",
                "ground_truth": {
                    "incident_type": "Vehicle Accident / Rescue",
                    "casualty_risk_level": 5,
                    "credibility_score": 0.94,
                    "location_text": "I-40 EB Mile 287 Cary exit",
                },
            },
            {
                "feed_id": "social_crisis",
                "title": "Citizen X: Total standstill I-40 East crash",
                "raw_text": "Avoid I-40 Eastbound by Cary exit completely stopped horrible rollover crash looks like 3 cars and 4 fire trucks helicopter landing",
                "ground_truth": {
                    "incident_type": "Vehicle Accident / Rescue",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.45,
                    "location_text": "I-40 Eastbound by Cary exit",
                },
            },
        ],
    },

    # ------------------ High-Impact Scenario 3: Flash Flood Submerged Vehicles ------------------
    {
        "scenario_id": "SCN-003-FLASH-FLOOD",
        "description": "Rapid urban flash flooding submerging vehicles near Crabtree Creek",
        "items": [
            {
                "feed_id": "weather_skywarn",
                "title": "SKYWARN REPORT: CRABTREE CREEK RAPID RISE",
                "raw_text": "FLASH FLOOD EMERGENCY: Crabtree Creek exceeding gauge 18.5ft, water rising 6in per 10min. Water across Atlantic Ave and Glenwood Ave low water bridges.",
                "ground_truth": {
                    "incident_type": "Severe Weather / Flooding",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.88,
                    "location_text": "Crabtree Creek, Atlantic Ave & Glenwood Ave",
                },
            },
            {
                "feed_id": "cad_police",
                "title": "CAD 911: WATER RESCUE ATLANTIC AVE NR CRABTREE",
                "raw_text": "WATER RESCUE: 2 vehicles stalled in swift water to windshields at Atlantic Ave & Hodges St nr Crabtree. Callers stranded on car roofs yelling for help.",
                "ground_truth": {
                    "incident_type": "Severe Weather / Flooding",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.92,
                    "location_text": "Atlantic Ave & Hodges St near Crabtree",
                },
            },
            {
                "feed_id": "fire_rescue",
                "title": "FIRE DISPATCH: SWIFTWATER BOAT TEAM 2 DEPLOYED",
                "raw_text": "SWIFTWATER 1 & 2 DISPATCHED: Atlantic Ave nr Crabtree Creek. 3 victims stranded on car roof in moving current. Water rescue technician entry.",
                "ground_truth": {
                    "incident_type": "Severe Weather / Flooding",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.95,
                    "location_text": "Atlantic Ave near Crabtree Creek",
                },
            },
        ],
    },

    # ------------------ High-Impact Scenario 4: Natural Gas Leak & Explosion Hazard ------------------
    {
        "scenario_id": "SCN-004-HAZMAT-GAS-LEAK",
        "description": "High pressure commercial gas main rupture downtown Raleigh",
        "items": [
            {
                "feed_id": "cad_police",
                "title": "CAD 911: 10-75 HAZMAT GAS LEAK 401 FAYETTEVILLE ST",
                "raw_text": "10-75 HAZMAT GAS LEAK 401 FAYETTEVILLE ST DOWNTOWN. CONSTRUCTION HIT 4-INCH HIGH PRESSURE GAS MAIN. ROARING SOUND STRONG ODOR EVACUATING OFFICE TOWER.",
                "ground_truth": {
                    "incident_type": "Infrastructure Failure / Gas Leak",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.91,
                    "location_text": "401 Fayetteville St, Downtown Raleigh",
                },
            },
            {
                "feed_id": "fire_rescue",
                "title": "FIRE HAZMAT: LEVEL 2 DISPATCH 401 FAYETTEVILLE",
                "raw_text": "HAZMAT DISPATCH: 401 Fayetteville St. High pressure gas line rupture, LEL meters reading 70% in lobby basement, explosion hazard, immediate 500ft perimeter.",
                "ground_truth": {
                    "incident_type": "Infrastructure Failure / Gas Leak",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.95,
                    "location_text": "401 Fayetteville St",
                },
            },
            {
                "feed_id": "social_crisis",
                "title": "Citizen X: Huge gas smell Fayetteville street sirens",
                "raw_text": "Everyone running out of 401 Fayetteville building cops screaming move back intense gas smell like rotten eggs sirens everywhere downtown",
                "ground_truth": {
                    "incident_type": "Infrastructure Failure / Gas Leak",
                    "casualty_risk_level": 3,
                    "credibility_score": 0.55,
                    "location_text": "401 Fayetteville St downtown",
                },
            },
        ],
    },

    # ------------------ Routine / Low-Risk Scenario 5: Routine Fender Bender ------------------
    {
        "scenario_id": "SCN-005-ROUTINE-FENDER-BENDER",
        "description": "Minor property damage only vehicle collision",
        "items": [
            {
                "feed_id": "cad_police",
                "title": "CAD 911: 10-50 PD CRABTREE VALLEY MALL LOT",
                "raw_text": "10-50 PD ONLY CRABTREE VALLEY MALL LOWER LOT NEAR SEARS. 2 VEHICLES NO INJURIES NO FLUID LEAKS DRIVERS EXCHANGING INFO REQ OFFICER FOR REPORT.",
                "ground_truth": {
                    "incident_type": "Vehicle Accident / Rescue",
                    "casualty_risk_level": 1,
                    "credibility_score": 0.85,
                    "location_text": "Crabtree Valley Mall, Raleigh",
                },
            },
        ],
    },

    # ------------------ Routine / Low-Risk Scenario 6: Accidental Fire Alarm ------------------
    {
        "scenario_id": "SCN-006-FALSE-ALARM",
        "description": "Waterflow maintenance alarm false trip",
        "items": [
            {
                "feed_id": "fire_rescue",
                "title": "FIRE OPS: COMMERCIAL WATERFLOW ALARM 500 S SALISBURY",
                "raw_text": "AUTOMATIC ALARM: 500 S Salisbury St. Zone 4 sprinkler riser flow. Building superintendent advises accidental trip during routine dry-pipe valve test. Cancel units.",
                "ground_truth": {
                    "incident_type": "False Alarm / Non-Emergency",
                    "casualty_risk_level": 1,
                    "credibility_score": 0.95,
                    "location_text": "500 S Salisbury St",
                },
            },
        ],
    },

    # ------------------ Scenario 7: Cardiac Arrest / Medical Emergency ------------------
    {
        "scenario_id": "SCN-007-CARDIAC-ARREST",
        "description": "Severe medical emergency in public facility",
        "items": [
            {
                "feed_id": "ems_scanner",
                "title": "EMS PRIORITY 1: CPR IN PROGRESS 3000 NEW BERN AVE",
                "raw_text": "ECHO RESPONSE: 3000 New Bern Ave nr lobby. 58yo male collapsed unresponsive CPR in progress AED advised shock, Medic 12 en route 2 mins.",
                "ground_truth": {
                    "incident_type": "Medical Emergency",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.93,
                    "location_text": "3000 New Bern Ave, Raleigh",
                },
            },
            {
                "feed_id": "cad_police",
                "title": "CAD 911: MEDICAL ASSIST 3000 NEW BERN AVE",
                "raw_text": "10-54 MEDICAL ASSIST 3000 NEW BERN AVE. OFFICER FIRST ON SCENE APPLYING AED CHEST COMPRESSIONS CPR IN PROGRESS.",
                "ground_truth": {
                    "incident_type": "Medical Emergency",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.91,
                    "location_text": "3000 New Bern Ave",
                },
            },
        ],
    },

    # ------------------ Scenario 8: Active Shooter / Threat Rumor vs Dispatch ------------------
    {
        "scenario_id": "SCN-008-ACTIVE-THREAT-REPORT",
        "description": "High tension active disturbance / potential threat",
        "items": [
            {
                "feed_id": "social_crisis",
                "title": "Citizen X: Gunshots heard near Glenwood South!",
                "raw_text": "I think I heard 5 gunshots outside the club on Glenwood Ave & Peace St everyone is running!! Police please hurry",
                "ground_truth": {
                    "incident_type": "Civil Unrest / Active Threat",
                    "casualty_risk_level": 4,
                    "credibility_score": 0.40,
                    "location_text": "Glenwood Ave & Peace St",
                },
            },
            {
                "feed_id": "cad_police",
                "title": "CAD 911: 10-33 SHOTS REPORTED GLENWOOD & PEACE",
                "raw_text": "10-33 CODE 3 SHOTS REPORTED GLENWOOD AVE & PEACE ST. CALLER REPORTS LOUD POPPING VEHICLE SPEEDING OFF. NO INJURIES REPORTED YET UNITS ARRIVING.",
                "ground_truth": {
                    "incident_type": "Civil Unrest / Active Threat",
                    "casualty_risk_level": 3,
                    "credibility_score": 0.75,
                    "location_text": "Glenwood Ave & Peace St, Raleigh",
                },
            },
        ],
    },
]


def get_feed_stream_items(feed_id: str) -> List[Dict[str, Any]]:
    """Return all items intended for a specific feed_id."""
    collected = []
    for scn in CRISIS_DISPATCH_SCENARIOS:
        for item in scn["items"]:
            if item["feed_id"] == feed_id:
                item_copy = dict(item)
                item_copy["scenario_id"] = scn["scenario_id"]
                collected.append(item_copy)
    return collected
