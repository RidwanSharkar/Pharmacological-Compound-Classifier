import pandas as pd
import ast


data_path = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundClassifications_scraped.csv'
data = pd.read_csv(data_path)

def safe_eval(str_list):
    try:
        return ast.literal_eval(str_list)
    except ValueError:
        return []

data['Activities'] = data['Activities'].apply(safe_eval)

#=======================================================================================================================

categories = {
    "Adenosine A1 Receptor Agonists",
    "Adenosine A1 Receptor Antagonists",
    "Adenosine A2 Receptor Agonists",
    "Adenosine A2 Receptor Antagonists",
    "Adenosine A3 Receptor Agonists",
    "Adenosine A3 Receptor Antagonists",

    #"Cholinesterase Inhibitors",
    #"Cholinesterase Reactivators",
    "Cholinergic Agents",
    "Cholinergic Agonists",
    "Cholinergic Antagonists",
    "Nicotinic Agonists",
    "Nicotinic Antagonists",
    "Muscarinic Agonists",
    "Muscarinic Antagonists",

    "Anti-Anxiety Agents",
    "Sleep Aids, Pharmaceutical",
    "Hypnotics and Sedatives",
    "Glycine Agents",

    "GABA Modulators",
    "GABA Agents",
    "GABA Agonists",
    "GABA Antagonists",
    "GABA Uptake Inhibitors",
    "GABA-A Receptor Agonists",
    "GABA-A Receptor Antagonists",
    "GABA-B Receptor Agonists",
    "GABA-B Receptor Antagonists",

    "Serotonin Agents",
    "Serotonin Receptor Agonists",
    "Serotonin Antagonists",
    "Serotonin 5-HT1 Receptor Agonists",
    "Serotonin 5-HT1 Receptor Antagonists",
    "Serotonin 5-HT2 Receptor Agonists",
    "Serotonin 5-HT2 Receptor Antagonists",
    "Serotonin 5-HT3 Receptor Agonists",
    "Serotonin 5-HT3 Receptor Antagonists",
    "Serotonin 5-HT4 Receptor Agonists",
    
    "Serotonin and Noradrenaline Reuptake Inhibitors",
    "Selective Serotonin Reuptake Inhibitors",
    "Antidepressive Agents",
    "Antidepressive Agents, Second-Generation",
    "Antidepressive Agents, Tricyclic",

    "Dopamine Agonists",
    "Dopamine Antagonists",
    "Dopamine Uptake Inhibitors",
    "Dopamine Agents",
    "Dopamine D2 Receptor Antagonists",
    "Antiparkinson Agents",

    "Central Nervous System Agents",
    "Central Nervous System Stimulants",
    "Muscle Relaxants, Central",
    "Central Nervous System Depressants",


    "Hallucinogens",
    "Psychotropic Drugs",
    "Antipsychotic Agents",
    "Sympathomimetics",
    "Sympatholytics",
    "Parasympathomimetics",
    "Antimanic Agents",
    "Antiemetics",
    "Adjuvants, Anesthesia",
    "Convulsants",

    "Cannabinoid Receptor Modulators",
    "Cannabinoid Receptor Agonists",
    "Cannabinoid Receptor Antagonists",

    #"Adjuvants, Pharmaceutic",

    "Nootropic Agents",
    "Neuroprotective Agents",
    "Neurotoxins",
    "Neurotransmitter Uptake Inhibitors",
    "Excitatory Amino Acid Agonists",
    "Excitatory Amino Acid Antagonists",
    "Neurotransmitter Agents",
    "Monoamine Oxidase Inhibitors",

    #"Voltage-Gated Sodium Channel Blockers",

    "Adrenergic Uptake Inhibitors",
    "Adrenergic Agents",
    "Adrenergic alpha-1 Receptor Agonists",
    "Adrenergic alpha-1 Receptor Antagonists",
    "Adrenergic alpha-2 Receptor Agonists",
    "Adrenergic alpha-2 Receptor Antagonists",
    "Adrenergic alpha-Agonists",
    "Adrenergic alpha-Antagonists",
    "Adrenergic Agonists",
    "Adrenergic beta-Agonists",
    "Adrenergic beta-Antagonists",
    "Adrenergic beta-1 Receptor Agonists",
    "Adrenergic beta-1 Receptor Antagonists",
    "Adrenergic beta-2 Receptor Agonists",
    "Adrenergic beta-2 Receptor Antagonists",
    "Adrenergic beta-3 Receptor Agonists",
    "Adrenergic beta-3 Receptor Antagonists",

    "Histamine H3 Antagonists",
    "Histamine Agonists",
    "Histamine H1 Antagonists, Non-Sedating",
    "Histamine Antagonists",
    "Histamine H1 Antagonists",
    "Histamine H2 Antagonists",
    
    "Analgesics",
    "Analgesics, Opioid",
    "Anesthetics, Local",
    "Analgesics, Non-Narcotic",

    "Narcotics",
    "Narcotic Antagonists",
    #"Smoking Cessation Agents",
    #"Anti-Obesity Agents",

    #"Peripheral Nervous System Agents",
    #"Wakefulness-Promoting Agents",
    #"Acetaldehyde Dehydrogenase Inhibitors",

    #"Purinergic P1 Receptor Agonists",
    "Purinergic P1 Receptor Antagonists",
    #"Purinergic P2 Receptor Agonists",
    #"Purinergic P2 Receptor Antagonists",
    #"Purinergic Agonists",
    #"Purinergic Antagonists",
    #"Purinergic P2X Receptor Antagonists",
    "Purinergic P2Y Receptor Antagonists"
}


filtered_data = data[data['Activities'].apply(lambda activities: any(category in activities for category in categories))]

cid_output_path = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\PsychoactiveCIDs.csv'
filtered_data['CID'].drop_duplicates().to_csv(cid_output_path, index=False)

print("CSV with CIDs has been created at:", cid_output_path)
