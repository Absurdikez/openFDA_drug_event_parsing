import glob
import numpy as np
import pandas as pd
from dask import delayed, compute
import dask.dataframe as dd
import pickle
import os

# Define directories
data_dir = "/Users/zekitopcu/Desktop/data_fda/openFDA_drug_event_parsing/ntbks/openFDA_drug_event_parsing/data/openFDA_drug_event/"
er_dir = data_dir + 'er_tables/'
dir_ = data_dir + 'report/'

# Check if the directory exists
print("Checking directory path:", dir_)
if not os.path.exists(dir_):
    print(f"Directory does not exist: {dir_}")
    raise FileNotFoundError(f"Directory not found: {dir_}")
else:
    # Use Dask to handle large number of files
    files = glob.glob(dir_ + '*.csv.gzip')
    print(f"Number of files found in {dir_}: {len(files)}")
    if len(files) == 0:
        print(f"No .csv.gz files found in the directory: {dir_}")

# Create the directory if it doesn't exist
try:
    os.mkdir(er_dir)
except FileExistsError:
    print(er_dir + " exists")

# In[ ]:

primarykey = 'safetyreportid'

def read_file(file):
    sample_df = pd.read_csv(file, compression='gzip', nrows=0) 
    # Dask DataFrame
    return dd.read_csv(
        file,
        compression='gzip',
        dtype={
            primarykey: 'str',
            'authoritynumb': 'object',
            'primarysource.literaturereference': 'object',
            'occurcountry': 'object',
            'primarysourcecountry': 'object',
            'receiver.receiverorganization': 'object',
            'reportduplicate.duplicatenumb': 'object',
            'reportduplicate.duplicatesource': 'object',
            'reporttype': 'object',
            'Unnamed: 0': 'object',
            'duplicate': 'object',
            'fulfillexpeditecriteria': 'object',
            'receiptdate': 'object',
            'receiptdateformat': 'object',
            'receivedate': 'object',
            'receivedateformat': 'object',
            'receiver.receivertype': 'object',
            'safetyreportversion': 'object',
            'sender.sendertype': 'object',
            'seriousnesscongenitalanomali': 'object',
            'seriousnessdeath': 'object',
            'seriousnessdisabling': 'object',
            'seriousnesshospitalization': 'object',
            'seriousnesslifethreatening': 'object',
            'seriousnessother': 'object',
            'transmissiondate': 'object',
            'transmissiondateformat': 'object',
            'reportduplicate': 'object',
            'primarysource.reportercountry': 'object',
            'companynumb': 'object',
            'receiver': 'object' 
        },
        assume_missing=True,
        blocksize=None
    ), sample_df
# ## ER tables

# ### report

# #### report_df

# In[ ]:

# Read files and combine using pandas
dir_ = data_dir + 'report/'
files = glob.glob(dir_ + '*.csv.gzip')
dfs = []

for file in files:
    df = pd.read_csv(file, compression='gzip', dtype={primarykey: 'str'})
    dfs.append(df)

# Combine using pandas
combined_df = pd.concat(dfs, sort=True)

# Convert to Dask DataFrame
report_df = dd.from_pandas(combined_df, npartitions=10)

# Continue with Dask
report_df[primarykey] = report_df[primarykey].astype(str)
print(report_df.columns.values)
report_df.head()
# #### report_er_df

# In[ ]:

columns = [primarykey, 'receiptdate', 'receivedate', 'transmissiondate']
rename_columns = {'receiptdate': 'mostrecent_receive_date',
                  'receivedate': 'receive_date',
                  'transmissiondate': 'lastupdate_date'}

report_er_df = (report_df[columns]
                .rename(columns=rename_columns)
                .set_index(primarykey)
                .map_partitions(lambda df: df.sort_index())
                .reset_index()
                .dropna(subset=[primarykey])
                .drop_duplicates()
               )
report_er_df = report_er_df.repartition(npartitions=10)
report_er_df[primarykey] = report_er_df[primarykey].astype(str)
report_er_df = report_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(report_er_df.info())
report_er_df.head()


# In[ ]:

# Save report_er_df to CSV using Dask
(report_er_df
 .groupby(primarykey)
 .agg(max)
 .reset_index()
 .dropna(subset=[primarykey])
 .to_csv(er_dir + 'report.csv.gz', compression='gzip', index=False, single_file=True)
)

# In[ ]:

del report_er_df

# ### report_serious

# In[ ]:

columns = [primarykey, 'serious', 'seriousnesscongenitalanomali',
           'seriousnesslifethreatening', 'seriousnessdisabling',
           'seriousnessdeath', 'seriousnessother']
rename_columns = {
    'seriousnesscongenitalanomali': 'congenital_anomali',
    'seriousnesslifethreatening': 'life_threatening',
    'seriousnessdisabling': 'disabling',
    'seriousnessdeath': 'death',
    'seriousnessother': 'other'}

report_serious_er_df = (report_df[columns]
                        .rename(columns=rename_columns)
                        .set_index(primarykey)
                        .map_partitions(lambda df: df.sort_index())
                        .reset_index()
                        .dropna(subset=[primarykey])
                        .drop_duplicates()
                        .groupby(primarykey)
                        .first()
                        .reset_index()
                        .dropna(subset=[primarykey])
                       )
report_serious_er_df[primarykey] = report_serious_er_df[primarykey].astype(str)
report_serious_er_df = report_serious_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(report_serious_er_df.info())
report_serious_er_df.head()

# In[ ]:

# Save report_serious_er_df to CSV using Dask
report_serious_er_df.to_csv(er_dir + 'report_serious.csv.gz', compression='gzip', index=False, single_file=True)

# ### reporter

# In[ ]:

columns = [primarykey, 'companynumb', 'primarysource.qualification', 'primarysource.reportercountry']
rename_columns = {'companynumb': 'reporter_company',
                  'primarysource.qualification': 'reporter_qualification',
                  'primarysource.reportercountry': 'reporter_country'}

reporter_er_df = (report_df[columns]
                  .rename(columns=rename_columns)
                  .set_index(primarykey)
                  .map_partitions(lambda df: df.sort_index())
                  .reset_index()
                  .dropna(subset=[primarykey])
                  .drop_duplicates()
                  .groupby(primarykey)
                  .first()
                  .reset_index()
                  .dropna(subset=[primarykey])
                 )
reporter_er_df[primarykey] = reporter_er_df[primarykey].astype(str)
reporter_er_df = reporter_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(reporter_er_df.info())
reporter_er_df.head()

# In[ ]:

# Save reporter_er_df to CSV using Dask
reporter_er_df.to_csv(er_dir + 'reporter.csv.gz', compression='gzip', index=False, single_file=True)

# In[ ]:

# Clean up dataframes
try:
    del df
except NameError:
    pass
try:
    del report_df
except NameError:
    pass
try:
    del report_serious_er_df
except NameError:
    pass
try:
    del report_er_df
except NameError:
    pass
try:
    del reporter_er_df
except NameError:
    pass

# ### patient

# #### patient_df

# In[ ]:

# Read files and combine using pandas
dir_ = data_dir + 'patient/'
files = glob.glob(dir_ + '*.csv.gzip')
dfs = []

for file in files:
    df = pd.read_csv(file, compression='gzip', dtype={primarykey: 'str'})
    dfs.append(df)

# Combine using pandas
combined_df = pd.concat(dfs, sort=True)

# Convert to Dask DataFrame
patient_df = dd.from_pandas(combined_df, npartitions=10)

# Continue with Dask
patient_df[primarykey] = patient_df[primarykey].astype(str)
print(patient_df.columns.values)
patient_df.head()

# #### patient_er_df

# In[ ]:

columns = [primarykey,
           'patient.patientonsetage',
           'patient.patientonsetageunit',
           'master_age',
           'patient.patientsex',
           'patient.patientweight']
rename_columns = {
    'patient.patientonsetage': 'patient_onsetage',
    'patient.patientonsetageunit': 'patient_onsetageunit',
    'master_age': 'patient_custom_master_age',
    'patient.patientsex': 'patient_sex',
    'patient.patientweight': 'patient_weight'
}

patient_er_df = (patient_df[columns]
                 .rename(columns=rename_columns)
                 .set_index(primarykey)
                 .map_partitions(lambda df: df.sort_index())
                 .reset_index()
                 .dropna(subset=[primarykey])
                 .drop_duplicates()
                 .groupby(primarykey)
                 .first()
                 .reset_index()
                 .dropna(subset=[primarykey])
                )
patient_er_df = patient_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(patient_er_df.info())
patient_er_df.head()

# In[ ]:

# Save patient_er_df to CSV using Dask
patient_er_df.to_csv(er_dir + 'patient.csv.gz', compression='gzip', index=False, single_file=True)

# In[ ]:

del patient_df

# ### drug_characteristics

# #### patient.drug

# In[ ]:

# Read files and combine using pandas
dir_ = data_dir + 'patient_drug/'
files = glob.glob(dir_ + '*.csv.gzip')
dfs = []

for file in files:
    df = pd.read_csv(file, compression='gzip', dtype={primarykey: 'str'})
    dfs.append(df)

# Combine using pandas
combined_df = pd.concat(dfs, sort=True)

# Convert to Dask DataFrame
patient_drug_df = dd.from_pandas(combined_df, npartitions=10)

# Continue with Dask
patient_drug_df[primarykey] = patient_drug_df[primarykey].astype(str)
print(patient_drug_df.columns.values)
patient_drug_df.head()

# In[ ]:

columns = [primarykey, 'medicinalproduct', 'drugcharacterization', 'drugadministrationroute', 'drugindication']
rename_columns = {
    'medicinalproduct': 'medicinal_product',
    'drugcharacterization': 'drug_characterization',
    'drugadministrationroute': 'drug_administration',
    'drugindication': 'drug_indication'
}

drugcharacteristics_er_df = (patient_drug_df[columns]
                             .rename(columns=rename_columns)
                             .set_index(primarykey)
                             .map_partitions(lambda df: df.sort_index())
                             .reset_index()
                             .drop_duplicates()
                             .dropna(subset=[primarykey])
                            )
drugcharacteristics_er_df = drugcharacteristics_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(drugcharacteristics_er_df.info())
drugcharacteristics_er_df.head()

# In[ ]:

# Save drugcharacteristics_er_df to CSV using Dask
drugcharacteristics_er_df.to_csv(er_dir + 'drugcharacteristics.csv.gz', compression='gzip', index=False, single_file=True)

# In[ ]:

del drugcharacteristics_er_df
del patient_drug_df

# ### drugs

# #### patient.drug.openfda.rxcui_df

# In[ ]:

dir_ = data_dir + 'patient_drug_openfda_rxcui/'
files = glob.glob(dir_ + '*.csv.gzip')
dfs = []

for file in files:
    df = pd.read_csv(file, compression='gzip', dtype={primarykey: 'str'})
    dfs.append(df)

# Combine using pandas
combined_df = pd.concat(dfs, sort=True)

# Convert to Dask DataFrame
patient_drug_openfda_rxcui_df = dd.from_pandas(combined_df, npartitions=10)

# Continue with Dask
print(patient_drug_openfda_rxcui_df.columns.values)
patient_drug_openfda_rxcui_df[primarykey] = patient_drug_openfda_rxcui_df[primarykey].astype(str)
patient_drug_openfda_rxcui_df['value'] = patient_drug_openfda_rxcui_df['value'].astype(int)
patient_drug_openfda_rxcui_df.head()

# #### drugs_er_df

# In[ ]:

columns = [primarykey, 'value']
rename_columns = {'value': 'rxcui'}

drugs_er_df = (patient_drug_openfda_rxcui_df[columns]
               .rename(columns=rename_columns)
               .set_index(primarykey)
               .map_partitions(lambda df: df.sort_index())
               .reset_index()
               .drop_duplicates()
               .dropna(subset=[primarykey])
              )
drugs_er_df = drugs_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(drugs_er_df.info())
drugs_er_df.head()

# In[ ]:

# Convert 'rxcui' column to integer
drugs_er_df['rxcui'] = drugs_er_df['rxcui'].astype(int)

# Convert primary key to string
drugs_er_df[primarykey] = drugs_er_df[primarykey].astype(str)

# Save drugs_er_df to CSV using Dask
drugs_er_df.to_csv(er_dir + 'drugs.csv.gz', compression='gzip', index=False, single_file=True)

# In[ ]:

del patient_drug_openfda_rxcui_df
del drugs_er_df

# ### reactions

# #### patient.reaction_df

# In[ ]:

dir_ = data_dir + 'patient_reaction/'
files = glob.glob(dir_ + '*.csv.gzip')
dfs = []

for file in files:
    df = pd.read_csv(file, compression='gzip', dtype={primarykey: 'str'})
    dfs.append(df)

# Combine using pandas
combined_df = pd.concat(dfs, sort=True)

# Convert to Dask DataFrame
patient_reaction_df = dd.from_pandas(combined_df, npartitions=10)

# Continue with Dask
patient_reaction_df[primarykey] = patient_reaction_df[primarykey].astype(str)
print(patient_reaction_df.columns.values)
patient_reaction_df.head()

# #### patient_reaction_er_df

# In[ ]:

columns = [primarykey, 'reactionmeddrapt', 'reactionoutcome']
rename_columns = {
    'reactionmeddrapt': 'reaction_meddrapt',
    'reactionoutcome': 'reaction_outcome'
}

reactions_er_df = (patient_reaction_df[columns]
                   .rename(columns=rename_columns)
                   .set_index(primarykey)
                   .map_partitions(lambda df: df.sort_index())
                   .reset_index()
                   .dropna(subset=[primarykey])
                   .drop_duplicates()
                  )
reactions_er_df[primarykey] = reactions_er_df[primarykey].astype(str)
reactions_er_df = reactions_er_df.map_partitions(lambda df: df.reindex(np.sort(df.columns), axis=1))
print(reactions_er_df.info())
reactions_er_df.head()

# In[ ]:

# Save reactions_er_df to CSV using Dask
reactions_er_df.to_csv(er_dir + 'reactions.csv.gz', compression='gzip', index=False, single_file=True)

# In[ ]:

del patient_reaction_df
del reactions_er_df

# ### omop tables for joining

# In[ ]:

# Read concept table using pandas
concept = pd.read_csv('../../vocabulary_SNOMED_MEDDRA_RxNorm_ATC/CONCEPT.csv', sep='\t',
                      dtype={'concept_id': 'int'})
concept.head()

# In[ ]:

# Read concept_relationship table using pandas
concept_relationship = pd.read_csv('../../vocabulary_SNOMED_MEDDRA_RxNorm_ATC/CONCEPT_RELATIONSHIP.csv', sep='\t',
                                   dtype={'concept_id_1': 'int', 'concept_id_2': 'int'})
concept_relationship.head()

# ### standard_drugs

# In[ ]:

# Read drugs table using Dask
drugs = dd.read_csv(er_dir + 'drugs.csv.gz', compression='gzip', dtype={'safetyreportid': 'str'}, blocksize=None)

# Convert 'rxcui' column to integer
drugs['rxcui'] = drugs['rxcui'].astype(int)

# Get unique rxcui values
urxcuis = drugs['rxcui'].unique().compute()

# Print the number of unique rxcui values and the first 5
print(len(urxcuis))
print(urxcuis[:5])

# Filter RxNorm concepts
rxnorm_concept = concept.query('vocabulary_id=="RxNorm"')

# Get unique concept codes
concept_codes = rxnorm_concept['concept_code'].astype(int).unique()
print(len(concept_codes))
print(len(urxcuis))

# Find intersection of concept codes and rxcui values
intersect = np.intersect1d(concept_codes, urxcuis)

print(len(intersect))
print(len(intersect) / len(urxcuis))

# Clean up
del urxcuis
del concept_codes

# Filter RxNorm concept IDs
rxnorm_concept_ids = (rxnorm_concept
                      .query('concept_code in @intersect')['concept_id']
                      .astype(int)
                      .unique()
                     )
all_rxnorm_concept_ids = rxnorm_concept['concept_id'].unique()

# Prepare concept relationship data
r = (concept_relationship
     .copy()
     .loc[:, ['concept_id_1', 'concept_id_2', 'relationship_id']]
     .drop_duplicates()
    )
c = rxnorm_concept.copy()
c['concept_id'] = c['concept_id'].astype(int)
c['concept_code'] = c['concept_code'].astype(int)

# Join drugs with RxNorm concepts
joined = (drugs
          .set_index('rxcui')
          .join(
              c.query('vocabulary_id=="RxNorm"')
              .loc[:, ['concept_id', 'concept_code', 'concept_name', 'concept_class_id']]
              .drop_duplicates()
              .set_index('concept_code')
          )
          .dropna()
          .rename_axis('RxNorm_concept_code')
          .reset_index()
          .rename(
              columns={
                  'concept_class_id': 'RxNorm_concept_class_id',
                  'concept_name': 'RxNorm_concept_name',
                  'concept_id': 'RxNorm_concept_id'
              }
          )
          .dropna(subset=['RxNorm_concept_id'])
          .drop_duplicates()
         )
joined = joined.reindex(np.sort(joined.columns), axis=1)
print(joined.shape)
print(joined.head())

# In[ ]:

# Calculate the intersection ratio
intersection_ratio = len(np.intersect1d(joined.RxNorm_concept_code.unique(), intersect)) / len(intersect)
print(intersection_ratio)

# Get unique RxNorm concept IDs
ids = joined.RxNorm_concept_id.dropna().astype(int).unique()

# Ensure the data directory exists
if not os.path.exists('../../data'):
    os.makedirs('../../data', exist_ok=True)

# Save the unique RxNorm concept IDs using pickle
pickle.dump(ids, open('../../data/all_openFDA_rxnorm_concept_ids.pkl', 'wb'))

# Save joined dataframe to CSV using Dask
joined.to_csv(er_dir + 'standard_drugs.csv.gz', compression='gzip', index=False, single_file=True)

# Clean up
del joined

# ### standard_reactions

# In[ ]:

# Read reactions table using Dask
patient_reaction_df = dd.read_csv(er_dir + 'reactions.csv.gz', compression='gzip', dtype={'safetyreportid': 'str'}, blocksize=None)
all_reports = patient_reaction_df.safetyreportid.unique().compute()
print(patient_reaction_df.columns)
print(patient_reaction_df.safetyreportid.nunique().compute())
print(patient_reaction_df.reaction_meddrapt.nunique().compute())

# Display the first few rows
patient_reaction_df.head()

# Filter MedDRA concepts
meddra_concept = concept.query('vocabulary_id=="MedDRA"')
meddra_concept.head()

# Get unique reaction names and concept names
reactions = patient_reaction_df.reaction_meddrapt.copy().astype(str).str.title().unique().compute()
print(len(reactions))
concept_names = meddra_concept.concept_name.astype(str).str.title().unique()
print(len(concept_names))

# Find intersection of reaction names and concept names
intersect_title = np.intersect1d(reactions, concept_names)
print(len(intersect_title))

# Calculate the intersection ratio
intersection_ratio_title = len(intersect_title) / len(reactions)
print(intersection_ratio_title)

# In[ ]:

# Convert 'reaction_meddrapt' to title case
patient_reaction_df['reaction_meddrapt'] = patient_reaction_df['reaction_meddrapt'].astype(str).str.title()
meddra_concept['concept_name'] = meddra_concept['concept_name'].astype(str).str.title()
print(patient_reaction_df.shape[0].compute())

# Join patient reactions with MedDRA concepts
joined = (patient_reaction_df
          .set_index('reaction_meddrapt')
          .join(
              meddra_concept
              .query('concept_class_id=="PT"')
              .loc[:, ['concept_id', 'concept_name', 'concept_code', 'concept_class_id']]
              .drop_duplicates()
              .set_index('concept_name')
          )
          .rename(columns={'concept_id': 'MedDRA_concept_id',
                           'concept_code': 'MedDRA_concept_code',
                           'concept_class_id': 'MedDRA_concept_class_id'})
          .drop_duplicates()
         ).rename_axis('MedDRA_concept_name').reset_index()
joined = joined.reindex(np.sort(joined.columns), axis=1)
print(joined.shape[0].compute())
print(joined.head())

# In[ ]:

del meddra_concept
del patient_reaction_df

# In[ ]:

# Filter non-null MedDRA concept IDs
joined_notnull = joined[joined.MedDRA_concept_id.notnull()]
print(joined_notnull.shape[0].compute())
joined_notnull['MedDRA_concept_id'] = joined_notnull['MedDRA_concept_id'].astype(int)
print(joined_notnull.head())

# Calculate intersection ratio
intersection_ratio_reports = len(
    np.intersect1d(
        all_reports,
        joined_notnull.safetyreportid.astype(str).unique().compute()
    )
) / len(all_reports)
print(intersection_ratio_reports)

# Print value counts and unique counts
print(joined_notnull.MedDRA_concept_class_id.value_counts().compute())
print(joined_notnull.safetyreportid.nunique().compute())
print(joined_notnull.MedDRA_concept_id.nunique().compute())

# Save unique MedDRA concept IDs using pickle
pickle.dump(
    joined_notnull.MedDRA_concept_id.astype(int).unique().compute(),
    open('../../data/all_openFDA_meddra_concept_ids.pkl', 'wb')
)

# Save joined_notnull dataframe to CSV using Dask
joined_notnull.to_csv(er_dir + 'standard_reactions.csv.gz', compression='gzip', index=False, single_file=True)

# Clean up
del joined_notnull
del joined

# In[ ]:

standard_drugs = dd.read_csv(er_dir + 'standard_drugs.csv.gz', compression='gzip', dtype={'safetyreportid': 'str'}, blocksize=None)

# Get unique safety report IDs from standard_drugs
all_reports = standard_drugs.safetyreportid.unique().compute()
print(len(all_reports))

# Convert RxNorm_concept_id to integer
standard_drugs['RxNorm_concept_id'] = standard_drugs['RxNorm_concept_id'].astype(int)

# Display the first few rows of standard_drugs
standard_drugs.head()

# Filter RxNorm concepts
rxnorm_concept = concept.query('vocabulary_id=="RxNorm"')
rxnorm_concept_ids = rxnorm_concept['concept_id'].unique()

# Get unique RxNorm concept IDs from standard_drugs
openfda_concept_ids = standard_drugs.RxNorm_concept_id.dropna().astype(int).unique()

# Filter ATC concepts
atc_concept = concept.query('vocabulary_id=="ATC" & concept_class_id=="ATC 5th"')

# Prepare concept relationship data
r = (concept_relationship
     .copy()
     .loc[:, ['concept_id_1', 'concept_id_2', 'relationship_id']]
     .drop_duplicates()
    )
r['concept_id_1'] = r['concept_id_1'].astype(int)
r['concept_id_2'] = r['concept_id_2'].astype(int)

# Prepare ATC and RxNorm concept data
ac = atc_concept.copy()
ac['concept_id'] = ac['concept_id'].astype(int)
atc_concept_ids = ac['concept_id'].unique()

rc = rxnorm_concept.copy()
rc['concept_id'] = rc['concept_id'].astype(int)
rxnorm_concept_ids = rc['concept_id'].unique()

# Join RxNorm to ATC relationships
rxnorm_to_atc_relationships = (r
                               .query('concept_id_1 in @openfda_concept_ids & concept_id_2 in @atc_concept_ids')
                               .set_index('concept_id_1')
                               .join(
                                   rc.loc[:, ['concept_id', 'concept_code', 'concept_name', 'concept_class_id']]
                                   .drop_duplicates()
                                   .set_index('concept_id')
                               )
                               .rename_axis('RxNorm_concept_id')
                               .reset_index()
                               .dropna()
                               .rename(
                                   columns={
                                       'concept_code': 'RxNorm_concept_code',
                                       'concept_class_id': 'RxNorm_concept_class_id',
                                       'concept_name': 'RxNorm_concept_name',
                                       'concept_id_2': 'ATC_concept_id',
                                   }
                               )
                               .set_index('ATC_concept_id')
                               .join(
                                   ac.loc[:, ['concept_id', 'concept_code', 'concept_name', 'concept_class_id']]
                                   .drop_duplicates()
                                   .set_index('concept_id')
                               )
                               .dropna()
                               .rename_axis('ATC_concept_id')
                               .reset_index()
                               .rename(
                                   columns={
                                       'concept_code': 'ATC_concept_code',
                                       'concept_class_id': 'ATC_concept_class_id',
                                       'concept_name': 'ATC_concept_name'
                                   }
                               )
                              )
rxnorm_to_atc_relationships['RxNorm_concept_id'] = rxnorm_to_atc_relationships['RxNorm_concept_id'].astype(int)
rxnorm_to_atc_relationships['ATC_concept_id'] = rxnorm_to_atc_relationships['ATC_concept_id'].astype(int)

rxnorm_to_atc_relationships = rxnorm_to_atc_relationships.reindex(np.sort(rxnorm_to_atc_relationships.columns), axis=1)
print(rxnorm_to_atc_relationships.shape)
print(rxnorm_to_atc_relationships.head())

# In[ ]:

# Display value counts for ATC_concept_class_id
rxnorm_to_atc_relationships.ATC_concept_class_id.value_counts().compute()

# In[ ]:

# Clean up temporary dataframes
del r
del ac
del rc

standard_drugs = (pd.read_csv(
    er_dir+'standard_drugs.csv.gz',
    compression='gzip',
    dtype={
        'safetyreportid' : 'str'
    }
))

# Join standard drugs with ATC relationships
standard_drugs_atc = (standard_drugs
                      .loc[:, ['RxNorm_concept_id', 'safetyreportid']]
                      .drop_duplicates()
                      .set_index('RxNorm_concept_id')
                      .join(rxnorm_to_atc_relationships.set_index('RxNorm_concept_id'))
                      .drop_duplicates()
                      .reset_index(drop=True)
                      .drop(['RxNorm_concept_code', 'RxNorm_concept_name',
                             'RxNorm_concept_class_id', 'relationship_id'], axis=1)
                      .dropna(subset=['ATC_concept_id'])
                      .drop_duplicates()
                     )

standard_drugs_atc = standard_drugs_atc.reindex(np.sort(standard_drugs_atc.columns), axis=1)
standard_drugs_atc['ATC_concept_id'] = standard_drugs_atc['ATC_concept_id'].astype(int)

# Calculate intersection ratio
intersection_ratio_atc = len(
    np.intersect1d(all_reports, standard_drugs_atc.safetyreportid.unique().compute())
) / len(all_reports)
print(intersection_ratio_atc)

# Display dataframe information
print(standard_drugs_atc.shape)
print(standard_drugs_atc.info())
print(standard_drugs_atc.head())

# In[ ]:

# Clean up temporary dataframes
del standard_drugs
del rxnorm_to_atc_relationships

# Save standard_drugs_atc dataframe to CSV using Dask
standard_drugs_atc.to_csv(er_dir + 'standard_drugs_atc.csv.gz', compression='gzip', index=False, single_file=True)

# Clean up
del standard_drugs_atc

# ### standard_drugs_rxnorm_ingredients

# In[ ]:

# Load all_openFDA_rxnorm_concept_ids using pickle
all_openFDA_rxnorm_concept_ids = pickle.load(open('../../data/all_openFDA_rxnorm_concept_ids.pkl', 'rb'))

# Display the loaded concept IDs
print(all_openFDA_rxnorm_concept_ids)

# Get all RxNorm concept IDs
all_rxnorm_concept_ids = concept.query('vocabulary_id=="RxNorm"').concept_id.astype(int).unique()

# Prepare concept relationship data
r = (concept_relationship
     .loc[:, ['concept_id_1', 'concept_id_2', 'relationship_id']]
     .drop_duplicates()
     .dropna()
     .copy()
    )
r['concept_id_1'] = r['concept_id_1'].astype(int)
r['concept_id_2'] = r['concept_id_2'].astype(int)

# Prepare RxNorm concept data
c = (concept
     .query('vocabulary_id=="RxNorm" & standard_concept=="S"')
     .loc[:, ['concept_id', 'concept_code', 'concept_class_id', 'concept_name']]
     .drop_duplicates()
     .dropna()
     .copy()
    )
c['concept_id'] = c['concept_id'].astype(int)

# Analyze RxNorm relationships
rxnorm_relationships = (r
                        .query('concept_id_1 in @all_rxnorm_concept_ids & ' +
                               'concept_id_2 in @all_rxnorm_concept_ids')
                        .relationship_id
                        .value_counts()
                       )
print(rxnorm_relationships)

# In[ ]:

# First to second relations
first_second_relations = (r
                          .query('concept_id_1 in @all_openFDA_rxnorm_concept_ids')
                          .set_index('concept_id_1')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_1': 'RxNorm_concept_id_1',
                                  'concept_code': 'RxNorm_concept_code_1',
                                  'concept_class_id': 'RxNorm_concept_class_id_1',
                                  'concept_name': 'RxNorm_concept_name_1'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_1')
                          .reset_index()
                          .set_index('concept_id_2')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_2': 'RxNorm_concept_id_2',
                                  'concept_code': 'RxNorm_concept_code_2',
                                  'concept_class_id': 'RxNorm_concept_class_id_2',
                                  'concept_name': 'RxNorm_concept_name_2',
                                  'relationship_id': 'relationship_id_12'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_2')
                          .reset_index()
                          .dropna()
                          .drop_duplicates()
                         )
first_second_relations = first_second_relations[
    first_second_relations.RxNorm_concept_id_1 != first_second_relations.RxNorm_concept_id_2
]
print(first_second_relations.shape)
first_second_relations = first_second_relations.reindex(np.sort(first_second_relations.columns), axis=1)
print(first_second_relations.head())

# In[ ]:

# Group by concept class IDs
(first_second_relations.loc[:, ['RxNorm_concept_class_id_1', 'RxNorm_concept_class_id_2']]
 .groupby(['RxNorm_concept_class_id_1', 'RxNorm_concept_class_id_2'])
 .count()
)

# In[ ]:

# Second to third relations
ids = first_second_relations.RxNorm_concept_id_2.astype(int).unique()

second_third_relations = (r
                          .query('concept_id_1 in @ids')
                          .set_index('concept_id_1')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_1': 'RxNorm_concept_id_2',
                                  'concept_code': 'RxNorm_concept_code_2',
                                  'concept_class_id': 'RxNorm_concept_class_id_2',
                                  'concept_name': 'RxNorm_concept_name_2'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_2')
                          .reset_index()
                          .set_index('concept_id_2')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_2': 'RxNorm_concept_id_3',
                                  'concept_code': 'RxNorm_concept_code_3',
                                  'concept_class_id': 'RxNorm_concept_class_id_3',
                                  'concept_name': 'RxNorm_concept_name_3',
                                  'relationship_id': 'relationship_id_23'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_3')
                          .reset_index()
                          .dropna()
                          .drop_duplicates()
                         )
second_third_relations = second_third_relations[
    second_third_relations.RxNorm_concept_id_2 != second_third_relations.RxNorm_concept_id_3
]
print(second_third_relations.shape)
second_third_relations = second_third_relations.reindex(np.sort(second_third_relations.columns), axis=1)
print(second_third_relations.head())

# In[ ]:

# Group by concept class IDs
(second_third_relations.loc[:, ['RxNorm_concept_class_id_2', 'RxNorm_concept_class_id_3']]
 .groupby(['RxNorm_concept_class_id_2', 'RxNorm_concept_class_id_3'])
 .count()
)

# In[ ]:

# Third to fourth relations
ids = second_third_relations.RxNorm_concept_id_3.astype(int).unique()

third_fourth_relations = (r
                          .query('concept_id_1 in @ids')
                          .set_index('concept_id_1')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_1': 'RxNorm_concept_id_3',
                                  'concept_code': 'RxNorm_concept_code_3',
                                  'concept_class_id': 'RxNorm_concept_class_id_3',
                                  'concept_name': 'RxNorm_concept_name_3'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_3')
                          .reset_index()
                          .set_index('concept_id_2')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_2': 'RxNorm_concept_id_4',
                                  'concept_code': 'RxNorm_concept_code_4',
                                  'concept_class_id': 'RxNorm_concept_class_id_4',
                                  'concept_name': 'RxNorm_concept_name_4',
                                  'relationship_id': 'relationship_id_34'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_4')
                          .reset_index()
                          .dropna()
                          .drop_duplicates()
                         )
third_fourth_relations = third_fourth_relations[
    third_fourth_relations.RxNorm_concept_id_3 != third_fourth_relations.RxNorm_concept_id_4
]
print(third_fourth_relations.shape)
third_fourth_relations = third_fourth_relations.reindex(np.sort(third_fourth_relations.columns), axis=1)
print(third_fourth_relations.head())

# In[ ]:

# Group by concept class IDs for third to fourth relations
(third_fourth_relations.loc[:, ['RxNorm_concept_class_id_3', 'RxNorm_concept_class_id_4']]
 .groupby(['RxNorm_concept_class_id_3', 'RxNorm_concept_class_id_4'])
 .count()
)

# In[ ]:

# Fourth to fifth relations
ids = third_fourth_relations.RxNorm_concept_id_4.astype(int).unique()

fourth_fifth_relations = (r
                          .query('concept_id_1 in @ids')
                          .set_index('concept_id_1')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_1': 'RxNorm_concept_id_4',
                                  'concept_code': 'RxNorm_concept_code_4',
                                  'concept_class_id': 'RxNorm_concept_class_id_4',
                                  'concept_name': 'RxNorm_concept_name_4'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_4')
                          .reset_index()
                          .set_index('concept_id_2')
                          .join(c.set_index('concept_id'))
                          .rename(
                              columns={
                                  'concept_id_2': 'RxNorm_concept_id_5',
                                  'concept_code': 'RxNorm_concept_code_5',
                                  'concept_class_id': 'RxNorm_concept_class_id_5',
                                  'concept_name': 'RxNorm_concept_name_5',
                                  'relationship_id': 'relationship_id_45'
                              }
                          )
                          .rename_axis('RxNorm_concept_id_5')
                          .reset_index()
                          .dropna()
                          .drop_duplicates()
                         )
fourth_fifth_relations = fourth_fifth_relations[
    fourth_fifth_relations.RxNorm_concept_id_4 != fourth_fifth_relations.RxNorm_concept_id_5
]
print(fourth_fifth_relations.shape)
fourth_fifth_relations = fourth_fifth_relations.reindex(np.sort(fourth_fifth_relations.columns), axis=1)
print(fourth_fifth_relations.head())

# In[ ]:

# Group by concept class IDs for fourth to fifth relations
(fourth_fifth_relations.loc[:, ['RxNorm_concept_class_id_4', 'RxNorm_concept_class_id_5']]
 .groupby(['RxNorm_concept_class_id_4', 'RxNorm_concept_class_id_5'])
 .count()
)

# In[ ]:

# Fifth to sixth relations
ids = fourth_fifth_relations.RxNorm_concept_id_5.astype(int).unique()

fifth_sixth_relations = (r
                         .query('concept_id_1 in @ids')
                         .set_index('concept_id_1')
                         .join(c.set_index('concept_id'))
                         .rename(
                             columns={
                                 'concept_id_1': 'RxNorm_concept_id_5',
                                 'concept_code': 'RxNorm_concept_code_5',
                                 'concept_class_id': 'RxNorm_concept_class_id_5',
                                 'concept_name': 'RxNorm_concept_name_5'
                             }
                         )
                         .rename_axis('RxNorm_concept_id_5')
                         .reset_index()
                         .set_index('concept_id_2')
                         .join(c.set_index('concept_id'))
                         .rename(
                             columns={
                                 'concept_id_2': 'RxNorm_concept_id_6',
                                 'concept_code': 'RxNorm_concept_code_6',
                                 'concept_class_id': 'RxNorm_concept_class_id_6',
                                 'concept_name': 'RxNorm_concept_name_6',
                                 'relationship_id': 'relationship_id_56'
                             }
                         )
                         .rename_axis('RxNorm_concept_id_6')
                         .reset_index()
                         .dropna()
                         .drop_duplicates()
                        )
fifth_sixth_relations = fifth_sixth_relations[
    fifth_sixth_relations.RxNorm_concept_id_5 != fifth_sixth_relations.RxNorm_concept_id_6
]
print(fifth_sixth_relations.shape)
fifth_sixth_relations = fifth_sixth_relations.reindex(np.sort(fifth_sixth_relations.columns), axis=1)
print(fifth_sixth_relations.head())

# In[ ]:

# Group by concept class IDs for fifth to sixth relations
(fifth_sixth_relations.loc[:, ['RxNorm_concept_class_id_5', 'RxNorm_concept_class_id_6']]
 .groupby(['RxNorm_concept_class_id_5', 'RxNorm_concept_class_id_6'])
 .count()
)

# In[ ]:


rxnorm_to_ings123 = (first_second_relations.
 set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
            'RxNorm_concept_name_2','RxNorm_concept_class_id_2']).
 join(second_third_relations.
      set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
                 'RxNorm_concept_name_2','RxNorm_concept_class_id_2'])
     ).
 query('RxNorm_concept_class_id_3=="Ingredient" & '+
       '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)').
                  reset_index()
)
print(rxnorm_to_ings123.shape)
print(rxnorm_to_ings123.head())


# In[ ]:


len(np.intersect1d(
    rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
    all_openFDA_rxnorm_concept_ids
))/len(all_openFDA_rxnorm_concept_ids)


# In[ ]:


(rxnorm_to_ings123.
loc[:,['RxNorm_concept_name_1','RxNorm_concept_name_3']].
drop_duplicates()
).head()


# In[ ]:


(rxnorm_to_ings123.
loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2',
       'RxNorm_concept_class_id_3']].
 drop_duplicates()
)


# In[ ]:


rxnorm_to_ings123_to_add = (rxnorm_to_ings123.
loc[:,['RxNorm_concept_id_1','RxNorm_concept_code_1',
       'RxNorm_concept_name_1','RxNorm_concept_class_id_1',
       'RxNorm_concept_id_3','RxNorm_concept_code_3',
       'RxNorm_concept_name_3','RxNorm_concept_class_id_3']].
 drop_duplicates().
 rename(
     columns={
         'RxNorm_concept_id_3' : 'RxNorm_concept_id_2',
         'RxNorm_concept_code_3' : 'RxNorm_concept_code_2',
         'RxNorm_concept_name_3' : 'RxNorm_concept_name_2',
         'RxNorm_concept_class_id_3' : 'RxNorm_concept_class_id_2'
     })
                            .drop_duplicates()
)
print(rxnorm_to_ings123_to_add.shape)
rxnorm_to_ings123_to_add.head()


# In[ ]:


rxnorm_to_ings1234 = (first_second_relations.
 set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
            'RxNorm_concept_name_2','RxNorm_concept_class_id_2']).
 join(second_third_relations.
      set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
                 'RxNorm_concept_name_2','RxNorm_concept_class_id_2'])
     ).
 query('RxNorm_concept_class_id_3!="Ingredient" & '+
       '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)').
                  reset_index().
                      set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                      ).
                      join(third_fourth_relations.
                          set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                          )
                          ).
 query('RxNorm_concept_class_id_4=="Ingredient"').
                      reset_index()
)
rxnorm_to_ings1234 = rxnorm_to_ings1234.reindex(np.sort(rxnorm_to_ings1234.columns),axis=1)
print(rxnorm_to_ings1234.shape)
rxnorm_to_ings1234.head()


# In[ ]:


(rxnorm_to_ings1234.
loc[:,['RxNorm_concept_name_1','RxNorm_concept_name_4']].
drop_duplicates()
).head()
len(np.intersect1d(rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique(),
                  all_openFDA_rxnorm_concept_ids
                  ))/len(all_openFDA_rxnorm_concept_ids)


# In[ ]:


(rxnorm_to_ings1234.
loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2',
       'RxNorm_concept_class_id_3','RxNorm_concept_class_id_4']].
 drop_duplicates()
)

# In[ ]:

# Prepare data to add for 1234 relations
rxnorm_to_ings1234_to_add = (rxnorm_to_ings1234
                             .loc[:, ['RxNorm_concept_id_1', 'RxNorm_concept_code_1',
                                      'RxNorm_concept_name_1', 'RxNorm_concept_class_id_1',
                                      'RxNorm_concept_id_4', 'RxNorm_concept_code_4',
                                      'RxNorm_concept_name_4', 'RxNorm_concept_class_id_4']]
                             .drop_duplicates()
                             .rename(
                                 columns={
                                     'RxNorm_concept_id_4': 'RxNorm_concept_id_2',
                                     'RxNorm_concept_code_4': 'RxNorm_concept_code_2',
                                     'RxNorm_concept_name_4': 'RxNorm_concept_name_2',
                                     'RxNorm_concept_class_id_4': 'RxNorm_concept_class_id_2'
                                 })
                             .drop_duplicates()
)
print(rxnorm_to_ings1234_to_add.shape)
rxnorm_to_ings1234_to_add.head()

# In[ ]:

# Calculate intersection ratio for combined ings123 and ings1234
intersection_ratio_combined = len(
    np.intersect1d(
        np.union1d(
            rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
            rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        all_openFDA_rxnorm_concept_ids
    )
) / len(all_openFDA_rxnorm_concept_ids)
print(intersection_ratio_combined)

# In[ ]:

# Join first-second, second-third, third-fourth, and fourth-fifth relations to find ingredients
rxnorm_to_ings12345 = (first_second_relations
                       .set_index(['RxNorm_concept_id_2', 'RxNorm_concept_code_2',
                                   'RxNorm_concept_name_2', 'RxNorm_concept_class_id_2'])
                       .join(second_third_relations
                             .set_index(['RxNorm_concept_id_2', 'RxNorm_concept_code_2',
                                         'RxNorm_concept_name_2', 'RxNorm_concept_class_id_2'])
                            )
                       .query('RxNorm_concept_class_id_3!="Ingredient" & ' +
                              '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)')
                       .reset_index()
                       .set_index(['RxNorm_concept_id_3', 'RxNorm_concept_code_3',
                                   'RxNorm_concept_name_3', 'RxNorm_concept_class_id_3'])
                       .join(third_fourth_relations
                             .set_index(['RxNorm_concept_id_3', 'RxNorm_concept_code_3',
                                         'RxNorm_concept_name_3', 'RxNorm_concept_class_id_3'])
                            )
                       .query('RxNorm_concept_class_id_4!="Ingredient" & ' +
                              '(RxNorm_concept_class_id_2!=RxNorm_concept_class_id_4)')
                       .reset_index()
                       .set_index(['RxNorm_concept_id_4', 'RxNorm_concept_code_4',
                                   'RxNorm_concept_name_4', 'RxNorm_concept_class_id_4'])
                       .join(fourth_fifth_relations
                             .set_index(['RxNorm_concept_id_4', 'RxNorm_concept_code_4',
                                         'RxNorm_concept_name_4', 'RxNorm_concept_class_id_4'])
                            )
                       .query('RxNorm_concept_class_id_5=="Ingredient"')
                       .reset_index()
)
rxnorm_to_ings12345 = rxnorm_to_ings12345.reindex(np.sort(rxnorm_to_ings12345.columns), axis=1)
print(rxnorm_to_ings12345.shape)
rxnorm_to_ings12345.head()

# In[ ]:

# Display unique concept names for 12345 relations
(rxnorm_to_ings12345
 .loc[:, ['RxNorm_concept_name_1', 'RxNorm_concept_name_5']]
 .drop_duplicates()
).head()

# Calculate intersection ratio for 12345 relations
intersection_ratio_ings12345 = len(np.intersect1d(
    rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique(),
    all_openFDA_rxnorm_concept_ids
)) / len(all_openFDA_rxnorm_concept_ids)
print(intersection_ratio_ings12345)

# Display unique concept class IDs for 12345 relations
(rxnorm_to_ings12345
 .loc[:, ['RxNorm_concept_class_id_1', 'RxNorm_concept_class_id_2',
          'RxNorm_concept_class_id_3', 'RxNorm_concept_class_id_4',
          'RxNorm_concept_class_id_5']]
 .drop_duplicates()
)

# In[ ]:

# Prepare data to add for 12345 relations
rxnorm_to_ings12345_to_add = (rxnorm_to_ings12345
                              .loc[:, ['RxNorm_concept_id_1', 'RxNorm_concept_code_1',
                                       'RxNorm_concept_name_1', 'RxNorm_concept_class_id_1',
                                       'RxNorm_concept_id_5', 'RxNorm_concept_code_5',
                                       'RxNorm_concept_name_5', 'RxNorm_concept_class_id_5']]
                              .drop_duplicates()
                              .rename(
                                  columns={
                                      'RxNorm_concept_id_5': 'RxNorm_concept_id_2',
                                      'RxNorm_concept_code_5': 'RxNorm_concept_code_2',
                                      'RxNorm_concept_name_5': 'RxNorm_concept_name_2',
                                      'RxNorm_concept_class_id_5': 'RxNorm_concept_class_id_2'
                                  })
                              .drop_duplicates()
)
print(rxnorm_to_ings12345_to_add.shape)
rxnorm_to_ings12345_to_add.head()

# In[ ]:

# Calculate intersection ratio for combined ings123, ings1234, and ings12345
intersection_ratio_combined_12345 = len(
    np.intersect1d(
        np.union1d(
            np.union1d(
                rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        all_openFDA_rxnorm_concept_ids
    )
) / len(all_openFDA_rxnorm_concept_ids)
print(intersection_ratio_combined_12345)

# In[ ]:

# Find set difference
set_difference = np.setdiff1d(
    all_openFDA_rxnorm_concept_ids,
    np.union1d(
        np.union1d(
            rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
            rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
    )
)
print(set_difference)

# In[ ]:

# Join first-second, second-third, third-fourth, fourth-fifth, and fifth-sixth relations to find ingredients
rxnorm_to_ings123456 = (first_second_relations
                        .set_index(['RxNorm_concept_id_2', 'RxNorm_concept_code_2',
                                    'RxNorm_concept_name_2', 'RxNorm_concept_class_id_2'])
                        .join(second_third_relations
                              .set_index(['RxNorm_concept_id_2', 'RxNorm_concept_code_2',
                                          'RxNorm_concept_name_2', 'RxNorm_concept_class_id_2'])
                             )
                        .query('RxNorm_concept_class_id_3!="Ingredient" & ' +
                               '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)')
                        .reset_index()
                        .set_index(['RxNorm_concept_id_3', 'RxNorm_concept_code_3',
                                    'RxNorm_concept_name_3', 'RxNorm_concept_class_id_3'])
                        .join(third_fourth_relations
                              .set_index(['RxNorm_concept_id_3', 'RxNorm_concept_code_3',
                                          'RxNorm_concept_name_3', 'RxNorm_concept_class_id_3'])
                             )
                        .query('RxNorm_concept_class_id_4!="Ingredient" & ' +
                               '(RxNorm_concept_class_id_2!=RxNorm_concept_class_id_4)')
                        .reset_index()
                        .set_index(['RxNorm_concept_id_4', 'RxNorm_concept_code_4',
                                    'RxNorm_concept_name_4', 'RxNorm_concept_class_id_4'])
                        .join(fourth_fifth_relations
                              .set_index(['RxNorm_concept_id_4', 'RxNorm_concept_code_4',
                                          'RxNorm_concept_name_4', 'RxNorm_concept_class_id_4'])
                             )
                        .query('RxNorm_concept_class_id_5!="Ingredient" & ' +
                               '(RxNorm_concept_class_id_3!=RxNorm_concept_class_id_5)')
                        .reset_index()
                        .set_index(['RxNorm_concept_id_5', 'RxNorm_concept_code_5',
                                    'RxNorm_concept_name_5', 'RxNorm_concept_class_id_5'])
                        .join(fifth_sixth_relations
                              .set_index(['RxNorm_concept_id_5', 'RxNorm_concept_code_5',
                                          'RxNorm_concept_name_5', 'RxNorm_concept_class_id_5'])
                             )
                        .query('RxNorm_concept_class_id_6=="Ingredient"')
                        .reset_index()
)
rxnorm_to_ings123456 = rxnorm_to_ings123456.reindex(np.sort(rxnorm_to_ings123456.columns), axis=1)
print(rxnorm_to_ings123456.shape)
rxnorm_to_ings123456.head()

# In[ ]:

# Display unique concept names for 123456 relations
(rxnorm_to_ings123456
 .loc[:, ['RxNorm_concept_name_1', 'RxNorm_concept_name_6']]
 .drop_duplicates()
).head()

# Calculate intersection ratio for 123456 relations
intersection_ratio_ings123456 = len(np.intersect1d(
    rxnorm_to_ings123456.RxNorm_concept_id_1.dropna().astype(int).unique(),
    all_openFDA_rxnorm_concept_ids
)) / len(all_openFDA_rxnorm_concept_ids)
print(intersection_ratio_ings123456)

# Display unique concept class IDs for 123456 relations
(rxnorm_to_ings123456
 .loc[:, ['RxNorm_concept_class_id_1', 'RxNorm_concept_class_id_2',
          'RxNorm_concept_class_id_3', 'RxNorm_concept_class_id_4',
          'RxNorm_concept_class_id_5', 'RxNorm_concept_class_id_6']]
 .drop_duplicates()
)

# In[ ]:

# Prepare data to add for 123456 relations
rxnorm_to_ings123456_to_add = (rxnorm_to_ings123456
                               .loc[:, ['RxNorm_concept_id_1', 'RxNorm_concept_code_1',
                                        'RxNorm_concept_name_1', 'RxNorm_concept_class_id_1',
                                        'RxNorm_concept_id_6', 'RxNorm_concept_code_6',
                                        'RxNorm_concept_name_6', 'RxNorm_concept_class_id_6']]
                               .drop_duplicates()
                               .rename(
                                   columns={
                                       'RxNorm_concept_id_6': 'RxNorm_concept_id_2',
                                       'RxNorm_concept_code_6': 'RxNorm_concept_code_2',
                                       'RxNorm_concept_name_6': 'RxNorm_concept_name_2',
                                       'RxNorm_concept_class_id_6': 'RxNorm_concept_class_id_2'
                                   })
                               .drop_duplicates()
)
print(rxnorm_to_ings123456_to_add.shape)
rxnorm_to_ings123456_to_add.head()

# In[ ]:

# Calculate final intersection ratio for all combined relations
final_intersection_ratio = len(
    np.intersect1d(
        np.union1d(
            np.union1d(
                np.union1d(
                    rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                    rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
                ),
                rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings123456.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        all_openFDA_rxnorm_concept_ids
    )
) / len(all_openFDA_rxnorm_concept_ids)
print(final_intersection_ratio)

# In[ ]:

# Find final set difference
final_set_difference = np.setdiff1d(
    all_openFDA_rxnorm_concept_ids,
    np.union1d(
        np.union1d(
            np.union1d(
                rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        rxnorm_to_ings123456.RxNorm_concept_id_1.dropna().astype(int).unique()
    )
)
print(final_set_difference)

# In[ ]:

# Concatenate all ingredient relations
rxnorm_to_ings_all = pd.concat(
    [
        rxnorm_to_ings123_to_add,
        rxnorm_to_ings1234_to_add,
        rxnorm_to_ings12345_to_add,
        rxnorm_to_ings123456_to_add
    ]
).dropna().drop_duplicates()
rxnorm_to_ings_all['RxNorm_concept_id_2'] = rxnorm_to_ings_all['RxNorm_concept_id_2'].astype(int)
print(rxnorm_to_ings_all.shape)
rxnorm_to_ings_all.head()

# In[ ]:

# Calculate intersection ratio for all ingredients
intersection_ratio_all_ings = len(
    np.intersect1d(
        rxnorm_to_ings_all.RxNorm_concept_id_1,
        all_openFDA_rxnorm_concept_ids
    )
) / len(all_openFDA_rxnorm_concept_ids)
print(intersection_ratio_all_ings)

# In[ ]:

# Read standard drugs data
standard_drug = (pd.read_csv(er_dir + 'standard_drugs.csv.gz',
                             compression='gzip',
                             dtype={'safetyreportid': 'str'})
                )
standard_drug['RxNorm_concept_id'] = standard_drug['RxNorm_concept_id'].astype(int)
all_reports = standard_drug.safetyreportid.astype(str).unique()
print(standard_drug.shape)
standard_drug.head()

# In[ ]:

# Join standard drugs with ingredients
standard_drug_ingredients = ((standard_drug
  .loc[:, ['RxNorm_concept_id', 'safetyreportid']]
  .drop_duplicates()
  .set_index(['RxNorm_concept_id'])
).join(rxnorm_to_ings_all
       .loc[:, ['RxNorm_concept_id_1', 'RxNorm_concept_id_2',
                'RxNorm_concept_code_2', 'RxNorm_concept_name_2',
                'RxNorm_concept_class_id_2']]
       .drop_duplicates()
       .set_index(['RxNorm_concept_id_1'])
).drop_duplicates()
 .rename(
     columns={
         'RxNorm_concept_id_2': 'RxNorm_concept_id',
         'RxNorm_concept_code_2': 'RxNorm_concept_code',
         'RxNorm_concept_name_2': 'RxNorm_concept_name',
         'RxNorm_concept_class_id_2': 'RxNorm_concept_class_id'
     })
 .reset_index(drop=True)
 .dropna()
 .drop_duplicates()
)
standard_drug_ingredients = (standard_drug_ingredients
                             .reindex(np.sort(standard_drug_ingredients.columns), axis=1)
                            )
print(standard_drug_ingredients.shape)
standard_drug_ingredients.head()

# In[ ]:

# Calculate intersection ratio for standard drug ingredients
intersection_ratio_standard_drug_ingredients = len(
    np.intersect1d(
        standard_drug_ingredients.safetyreportid.astype(str).unique(),
        all_reports
    )
) / len(all_reports)
print(intersection_ratio_standard_drug_ingredients)

# In[ ]:

# Save standard drug ingredients to CSV
standard_drug_ingredients.to_csv(er_dir + 'standard_drugs_rxnorm_ingredients.csv.gz', compression='gzip', index=False)

# In[ ]:

# Read standard reactions data
standard_reactions = (pd.read_csv(er_dir + 'standard_reactions.csv.gz',
                                  compression="gzip",
                                  dtype={'safetyreportid': 'str'})
                     )
all_reports = standard_reactions.safetyreportid.unique()
print(standard_reactions.shape)
print(standard_reactions.head())

# In[ ]:

# Get unique MedDRA concept IDs from standard reactions
reactions = standard_reactions.MedDRA_concept_id.astype(int).unique()
print(len(reactions))

# Get all MedDRA concept IDs
meddra_concept_ids = concept.query('vocabulary_id=="MedDRA"').concept_id.astype(int).unique()
len(meddra_concept_ids)

# Calculate intersection
intersect = np.intersect1d(reactions, meddra_concept_ids)
print(len(intersect))
print(len(intersect) / len(reactions))

# In[ ]:

# Prepare MedDRA concept data
meddra_concept = concept.query('vocabulary_id=="MedDRA"')
meddra_concept['concept_id'] = meddra_concept.concept_id.astype(int)
all_meddra_concept_ids = meddra_concept.concept_id.unique()

# Prepare concept relationship data
r = (concept_relationship
     .copy()
     .loc[:, ['concept_id_1', 'concept_id_2', 'relationship_id']]
     .drop_duplicates()
    )
r['concept_id_1'] = r['concept_id_1'].astype(int)
r['concept_id_2'] = r['concept_id_2'].astype(int)

# In[ ]:

# Analyze MedDRA relationships
relationship_counts = (r
                       .query('concept_id_1 in @all_meddra_concept_ids & ' +
                              'concept_id_2 in @all_meddra_concept_ids')
                       .relationship_id.value_counts()
                      )
print(relationship_counts)

# In[ ]:

# Prepare all MedDRA relationships
c = meddra_concept.copy()

all_meddra_relationships = (r
                            .query('concept_id_1 in @meddra_concept_ids & ' +
                                   'concept_id_2 in @meddra_concept_ids')
                            .set_index('concept_id_1')
                            .join(
                                c.query('vocabulary_id=="MedDRA"')
                                .loc[:, ['concept_id', 'concept_code', 'concept_name', 'concept_class_id']]
                                .drop_duplicates()
                                .set_index('concept_id')
                            )
                            .rename_axis('MedDRA_concept_id_1')
                            .reset_index()
                            .rename(
                                columns={
                                    'concept_code': 'MedDRA_concept_code_1',
                                    'concept_class_id': 'MedDRA_concept_class_id_1',
                                    'concept_name': 'MedDRA_concept_name_1',
                                    'concept_id_2': 'MedDRA_concept_id_2',
                                    'relationship_id': 'relationship_id_12'
                                }
                            )
                            .set_index('MedDRA_concept_id_2')
                            .join(
                                c.query('vocabulary_id=="MedDRA"')
                                .loc[:, ['concept_id', 'concept_code', 'concept_name', 'concept_class_id']]
                                .drop_duplicates()
                                .set_index('concept_id')
                            )
                            .rename_axis('MedDRA_concept_id_2')
                            .reset_index()
                            .rename(
                                columns={
                                    'concept_code': 'MedDRA_concept_code_2',
                                    'concept_class_id': 'MedDRA_concept_class_id_2',
                                    'concept_name': 'MedDRA_concept_name_2'
                                }
                            )
)
all_meddra_relationships = all_meddra_relationships.reindex(np.sort(all_meddra_relationships.columns), axis=1)
print(all_meddra_relationships.shape)
print(all_meddra_relationships.head())

# In[ ]:

# Display value counts for MedDRA concept class IDs
print(all_meddra_relationships.MedDRA_concept_class_id_1.value_counts())
print(all_meddra_relationships.MedDRA_concept_class_id_2.value_counts())

# In[ ]:

# Convert MedDRA concept IDs and codes to integers
all_meddra_relationships['MedDRA_concept_id_1'] = all_meddra_relationships['MedDRA_concept_id_1'].astype(int)
all_meddra_relationships['MedDRA_concept_code_1'] = all_meddra_relationships['MedDRA_concept_code_1'].astype(int)
all_meddra_relationships['MedDRA_concept_id_2'] = all_meddra_relationships['MedDRA_concept_id_2'].astype(int)
all_meddra_relationships['MedDRA_concept_code_2'] = all_meddra_relationships['MedDRA_concept_code_2'].astype(int)

# In[ ]:

# Find first level relations
first_rxs = reactions
first_relations = (all_meddra_relationships
                   .query('MedDRA_concept_id_1 in @first_rxs & ' +
                          'MedDRA_concept_class_id_2=="HLT"')
                  ).reset_index(drop=True)
first_relations = first_relations[
    first_relations.MedDRA_concept_id_1 != first_relations.MedDRA_concept_id_2
]
print(first_relations.shape)
print(first_relations.head())
print(first_relations.MedDRA_concept_class_id_2.value_counts())

# In[ ]:

# Find second level relations
second_rxs = first_relations.MedDRA_concept_id_2.unique()
second_relations = (all_meddra_relationships
                    .query('MedDRA_concept_id_1 in @second_rxs & ' +
                           'MedDRA_concept_class_id_2=="HLGT"')
                    .rename(columns={
                        'MedDRA_concept_id_2': 'MedDRA_concept_id_3',
                        'MedDRA_concept_code_2': 'MedDRA_concept_code_3',
                        'MedDRA_concept_name_2': 'MedDRA_concept_name_3',
                        'MedDRA_concept_class_id_2': 'MedDRA_concept_class_id_3',
                        'MedDRA_concept_id_1': 'MedDRA_concept_id_2',
                        'MedDRA_concept_code_1': 'MedDRA_concept_code_2',
                        'MedDRA_concept_name_1': 'MedDRA_concept_name_2',
                        'MedDRA_concept_class_id_1': 'MedDRA_concept_class_id_2',
                        'relationship_id_12': 'relationship_id_23'
                    })
                  ).reset_index(drop=True)
second_relations = second_relations[
    second_relations.MedDRA_concept_id_2 != second_relations.MedDRA_concept_id_3
]
print(second_relations.shape)
print(second_relations.head())
print(second_relations.MedDRA_concept_class_id_2.value_counts())
print(second_relations.MedDRA_concept_class_id_3.value_counts())

# In[ ]:

# Find third level relations
third_rxs = second_relations.MedDRA_concept_id_3.unique()
third_relations = (all_meddra_relationships
                   .query('MedDRA_concept_id_1 in @third_rxs & ' +
                          'MedDRA_concept_class_id_2=="SOC"')
                   .rename(columns={
                       'MedDRA_concept_id_2': 'MedDRA_concept_id_4',
                       'MedDRA_concept_code_2': 'MedDRA_concept_code_4',
                       'MedDRA_concept_name_2': 'MedDRA_concept_name_4',
                       'MedDRA_concept_class_id_2': 'MedDRA_concept_class_id_4',
                       'MedDRA_concept_id_1': 'MedDRA_concept_id_3',
                       'MedDRA_concept_code_1': 'MedDRA_concept_code_3',
                       'MedDRA_concept_name_1': 'MedDRA_concept_name_3',
                       'MedDRA_concept_class_id_1': 'MedDRA_concept_class_id_3',
                       'relationship_id_12': 'relationship_id_34'
                   })
                  ).reset_index(drop=True)
third_relations = third_relations[
    third_relations.MedDRA_concept_id_3 != third_relations.MedDRA_concept_id_4
]
print(third_relations.shape)
print(third_relations.head())
print(third_relations.MedDRA_concept_class_id_3.value_counts())
print(third_relations.MedDRA_concept_class_id_4.value_counts())

# In[ ]:

# Combine first, second, and third level relations
first_second_third_relations = (first_relations
                                .set_index('MedDRA_concept_id_2')
                                .join(second_relations
                                      .loc[:, ['MedDRA_concept_id_2', 'MedDRA_concept_id_3',
                                               'MedDRA_concept_name_3', 'MedDRA_concept_class_id_3',
                                               'MedDRA_concept_code_3', 'relationship_id_23']]
                                      .set_index('MedDRA_concept_id_2')
                                     )
                                .reset_index()
)
first_second_third_relations = first_second_third_relations.reindex(np.sort(first_second_third_relations.columns), axis=1)
first_second_third_relations['MedDRA_concept_id_3'] = first_second_third_relations['MedDRA_concept_id_3'].astype(int)
print(first_second_third_relations.shape)
print(first_second_third_relations.head())
print(first_second_third_relations.MedDRA_concept_class_id_1.value_counts())
print(first_second_third_relations.MedDRA_concept_class_id_2.value_counts())
print(first_second_third_relations.MedDRA_concept_class_id_3.value_counts())

# In[ ]:

# Combine first, second, third, and fourth level relations
first_second_third_fourth_relations = (first_relations
                                       .set_index('MedDRA_concept_id_2')
                                       .join(second_relations
                                             .loc[:, ['MedDRA_concept_id_2', 'MedDRA_concept_id_3',
                                                      'MedDRA_concept_name_3', 'MedDRA_concept_class_id_3',
                                                      'MedDRA_concept_code_3', 'relationship_id_23']]
                                             .drop_duplicates()
                                             .set_index('MedDRA_concept_id_2')
                                            )
                                       .reset_index()
                                       .set_index('MedDRA_concept_id_3')
                                       .join(third_relations
                                             .loc[:, ['MedDRA_concept_id_3', 'MedDRA_concept_id_4',
                                                      'MedDRA_concept_name_4', 'MedDRA_concept_class_id_4',
                                                      'MedDRA_concept_code_4', 'relationship_id_34']]
                                             .drop_duplicates()
                                             .set_index('MedDRA_concept_id_3')
                                            )
                                       .reset_index()
)
first_second_third_fourth_relations = first_second_third_fourth_relations.reindex(np.sort(first_second_third_fourth_relations.columns), axis=1)
first_second_third_fourth_relations['MedDRA_concept_id_4'] = first_second_third_fourth_relations['MedDRA_concept_id_4'].astype(int)
print(first_second_third_fourth_relations.shape)
print(first_second_third_fourth_relations.head())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_1.value_counts())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_2.value_counts())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_3.value_counts())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_4.value_counts())

# In[ ]:

# Calculate the number of reactions not in the combined relations
leftover_count = len(np.setdiff1d(reactions, first_second_third_fourth_relations.MedDRA_concept_id_1.unique()))
print(leftover_count)

# In[ ]:

# Find leftover reactions
left_over = np.setdiff1d(reactions, first_second_third_fourth_relations.MedDRA_concept_id_1.unique())
leftover_relationships = all_meddra_relationships.query('MedDRA_concept_id_1 in @left_over')
print(leftover_relationships)

# In[ ]:

# Prepare df1 with unique MedDRA concept IDs from standard reactions
df1 = (standard_reactions
       .loc[:, ['MedDRA_concept_id']]
       .drop_duplicates()
       .dropna()
       .set_index('MedDRA_concept_id')
      )
print(df1.shape)

# In[ ]:

# Prepare df2 with first-second-third-fourth relations
df2 = (first_second_third_fourth_relations
       .set_index('MedDRA_concept_id_1')
      )
print(df2.shape)

# In[ ]:

# Join df1 and df2
joined = df1.join(df2).rename_axis('MedDRA_concept_id_1').reset_index().dropna()
joined = joined.reindex(np.sort(joined.columns), axis=1)
joined['MedDRA_concept_id_1'] = joined['MedDRA_concept_id_1'].astype(int).copy()
joined['MedDRA_concept_id_2'] = joined['MedDRA_concept_id_2'].astype(int).copy()
joined['MedDRA_concept_id_3'] = joined['MedDRA_concept_id_3'].astype(int).copy()
joined['MedDRA_concept_id_4'] = joined['MedDRA_concept_id_4'].astype(int).copy()
joined['MedDRA_concept_code_1'] = joined['MedDRA_concept_code_1'].astype(int).copy()
joined['MedDRA_concept_code_2'] = joined['MedDRA_concept_code_2'].astype(int).copy()
joined['MedDRA_concept_code_3'] = joined['MedDRA_concept_code_3'].astype(int).copy()
joined['MedDRA_concept_code_4'] = joined['MedDRA_concept_code_4'].astype(int).copy()
print(joined.shape)
print(joined.head())

# In[ ]:

# Display value counts for MedDRA concept class IDs in joined data
print(joined.MedDRA_concept_class_id_1.value_counts())
print(joined.MedDRA_concept_class_id_2.value_counts())
print(joined.MedDRA_concept_class_id_3.value_counts())
print(joined.MedDRA_concept_class_id_4.value_counts())

# In[ ]:

# Save joined data to CSV
joined.to_csv(er_dir + 'standard_reactions_meddra_relationships.csv.gz', compression='gzip', index=False)

# In[ ]:

# Extract PT to SOC relationships
pt_to_soc = (joined
             .loc[:, ['MedDRA_concept_id_1', 'MedDRA_concept_code_1',
                      'MedDRA_concept_name_1', 'MedDRA_concept_class_id_1',
                      'MedDRA_concept_id_4', 'MedDRA_concept_code_4',
                      'MedDRA_concept_name_4', 'MedDRA_concept_class_id_4']]
             .query('MedDRA_concept_class_id_4=="SOC"')
             .drop_duplicates()
)
print(pt_to_soc.shape)
print(pt_to_soc.head())

# In[ ]:

# Extract PT to HLGT relationships
pt_to_hlgt = (joined
              .loc[:, ['MedDRA_concept_id_1', 'MedDRA_concept_code_1',
                       'MedDRA_concept_name_1', 'MedDRA_concept_class_id_1',
                       'MedDRA_concept_id_3', 'MedDRA_concept_code_3',
                       'MedDRA_concept_name_3', 'MedDRA_concept_class_id_3']]
              .query('MedDRA_concept_class_id_3=="HLGT"')
              .drop_duplicates()
)
print(pt_to_hlgt.shape)
print(pt_to_hlgt.head())

# In[ ]:

# Extract PT to HLT relationships
pt_to_hlt = (joined
             .loc[:, ['MedDRA_concept_id_1', 'MedDRA_concept_code_1',
                      'MedDRA_concept_name_1', 'MedDRA_concept_class_id_1',
                      'MedDRA_concept_id_2', 'MedDRA_concept_code_2',
                      'MedDRA_concept_name_2', 'MedDRA_concept_class_id_2']]
             .query('MedDRA_concept_class_id_2=="HLT"')
             .drop_duplicates()
)
print(pt_to_hlt.shape)
print(pt_to_hlt.head())

# In[ ]:

# Join standard reactions with PT to HLT relationships
standard_reactions_pt_to_hlt = (standard_reactions
                                .loc[:, ['safetyreportid', 'MedDRA_concept_id']]
                                .drop_duplicates()
                                .set_index(['MedDRA_concept_id'])
                                .join(pt_to_hlt
                                      .loc[:, ['MedDRA_concept_id_1', 'MedDRA_concept_id_2',
                                               'MedDRA_concept_code_2', 'MedDRA_concept_name_2',
                                               'MedDRA_concept_class_id_2']]
                                      .set_index('MedDRA_concept_id_1')
                                     )
                                .reset_index(drop=True)
                                .rename(
                                    columns={
                                        'MedDRA_concept_id_2': 'MedDRA_concept_id',
                                        'MedDRA_concept_code_2': 'MedDRA_concept_code',
                                        'MedDRA_concept_name_2': 'MedDRA_concept_name',
                                        'MedDRA_concept_class_id_2': 'MedDRA_concept_class_id'
                                    }
                                )
                                .dropna()
                                .drop_duplicates()
)
standard_reactions_pt_to_hlt = standard_reactions_pt_to_hlt.reindex(np.sort(standard_reactions_pt_to_hlt.columns), axis=1)
print(standard_reactions_pt_to_hlt.shape)
print(standard_reactions_pt_to_hlt.head())

# In[ ]:

# Calculate intersection ratio for PT to HLT
intersection_ratio_hlt = len(
    np.intersect1d(
        all_reports,
        standard_reactions_pt_to_hlt.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_hlt)

# In[ ]:

# Save PT to HLT data to CSV
standard_reactions_pt_to_hlt.to_csv(er_dir + 'standard_reactions_meddra_hlt.csv.gz', compression='gzip', index=False)

# In[ ]:

# Join standard reactions with PT to HLGT relationships
standard_reactions_pt_to_hlgt = (standard_reactions
                                 .loc[:, ['safetyreportid', 'MedDRA_concept_id']]
                                 .drop_duplicates()
                                 .set_index(['MedDRA_concept_id'])
                                 .join(pt_to_hlgt
                                       .loc[:, ['MedDRA_concept_id_1', 'MedDRA_concept_id_3',
                                                'MedDRA_concept_code_3', 'MedDRA_concept_name_3',
                                                'MedDRA_concept_class_id_3']]
                                       .set_index('MedDRA_concept_id_1')
                                      )
                                 .reset_index(drop=True)
                                 .rename(
                                     columns={
                                         'MedDRA_concept_id_3': 'MedDRA_concept_id',
                                         'MedDRA_concept_code_3': 'MedDRA_concept_code',
                                         'MedDRA_concept_name_3': 'MedDRA_concept_name',
                                         'MedDRA_concept_class_id_3': 'MedDRA_concept_class_id'
                                     }
                                 )
                                 .dropna()
                                 .drop_duplicates()
)
standard_reactions_pt_to_hlgt = standard_reactions_pt_to_hlgt.reindex(np.sort(standard_reactions_pt_to_hlgt.columns), axis=1)
print(standard_reactions_pt_to_hlgt.shape)
print(standard_reactions_pt_to_hlgt.head())

# In[ ]:

# Calculate intersection ratio for PT to HLGT
intersection_ratio_hlgt = len(
    np.intersect1d(
        all_reports,
        standard_reactions_pt_to_hlgt.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_hlgt)

# In[ ]:

# Save PT to HLGT data to CSV
standard_reactions_pt_to_hlgt.to_csv(er_dir + 'standard_reactions_meddra_hlgt.csv.gz', compression='gzip', index=False)

# In[ ]:

# Join standard reactions with PT to SOC relationships
standard_reactions_pt_to_soc = (standard_reactions
                                .loc[:, ['safetyreportid', 'MedDRA_concept_id']]
                                .drop_duplicates()
                                .set_index(['MedDRA_concept_id'])
                                .join(pt_to_soc
                                      .loc[:, ['MedDRA_concept_id_1', 'MedDRA_concept_id_4',
                                               'MedDRA_concept_code_4', 'MedDRA_concept_name_4',
                                               'MedDRA_concept_class_id_4']]
                                      .set_index('MedDRA_concept_id_1')
                                     )
                                .reset_index(drop=True)
                                .rename(
                                    columns={
                                        'MedDRA_concept_id_4': 'MedDRA_concept_id',
                                        'MedDRA_concept_code_4': 'MedDRA_concept_code',
                                        'MedDRA_concept_name_4': 'MedDRA_concept_name',
                                        'MedDRA_concept_class_id_4': 'MedDRA_concept_class_id'
                                    }
                                )
                                .dropna()
                                .drop_duplicates()
)
standard_reactions_pt_to_soc = standard_reactions_pt_to_soc.reindex(np.sort(standard_reactions_pt_to_soc.columns), axis=1)
print(standard_reactions_pt_to_soc.shape)
print(standard_reactions_pt_to_soc.head())

# In[ ]:

# Calculate intersection ratio for PT to SOC
intersection_ratio_soc = len(
    np.intersect1d(
        all_reports,
        standard_reactions_pt_to_soc.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_soc)

# In[ ]:

# Save PT to SOC data to CSV
standard_reactions_pt_to_soc.to_csv(er_dir + 'standard_reactions_meddra_soc.csv.gz', compression='gzip', index=False)

# In[ ]:

# Clean up
del c
del r
del first_relations
del second_relations
del first_second_third_relations
del all_meddra_relationships
del meddra_concept
del df1
del df2
del joined
del standard_reactions_pt_to_soc
del standard_reactions_pt_to_hlgt
del standard_reactions_pt_to_hlt

# In[ ]:

# Read MedDRA relationships data
standard_reactions_meddra_relationships = pd.read_csv(
    er_dir + 'standard_reactions_meddra_relationships.csv.gz',
    compression='gzip',
    dtype={'safetyreportid': 'str'}
)

print(standard_reactions_meddra_relationships.MedDRA_concept_id_1.nunique())
print(standard_reactions_meddra_relationships.MedDRA_concept_id_2.nunique())
print(standard_reactions_meddra_relationships.MedDRA_concept_id_3.nunique())
print(standard_reactions_meddra_relationships.MedDRA_concept_id_4.nunique())

standard_reactions_meddra_relationships['MedDRA_concept_id_1'] = standard_reactions_meddra_relationships['MedDRA_concept_id_1'].astype(int)
standard_reactions_meddra_relationships['MedDRA_concept_id_2'] = standard_reactions_meddra_relationships['MedDRA_concept_id_2'].astype(int)
standard_reactions_meddra_relationships['MedDRA_concept_id_3'] = standard_reactions_meddra_relationships['MedDRA_concept_id_3'].astype(int)
standard_reactions_meddra_relationships['MedDRA_concept_id_4'] = standard_reactions_meddra_relationships['MedDRA_concept_id_4'].astype(int)

print(standard_reactions_meddra_relationships.shape)
print(standard_reactions_meddra_relationships.head())

# In[ ]:

# Calculate intersection with MedDRA concept IDs
reactions = standard_reactions_meddra_relationships.MedDRA_concept_id_1.unique()
print(len(reactions))
meddra_concept_ids = concept.query('vocabulary_id=="MedDRA"').concept_id.astype(int).unique()
len(meddra_concept_ids)

intersect = np.intersect1d(reactions, meddra_concept_ids)
print(len(intersect))
print(len(intersect) / len(reactions))

# In[ ]:

# Map MedDRA to SNOMED relationships
m_to_s_r = (concept_relationship
            .query('relationship_id=="MedDRA - SNOMED eq"')
            .loc[:, ['concept_id_1', 'concept_id_2']]
            .drop_duplicates()
            .set_index('concept_id_2')
            .join(concept
                  .query('vocabulary_id=="SNOMED"')
                  .loc[:, ['concept_id', 'concept_code', 'concept_class_id', 'concept_name']]
                  .drop_duplicates()
                  .set_index('concept_id')
                 )
            .rename_axis('SNOMED_concept_id')
            .reset_index()
            .rename(columns={
                'concept_id_1': 'MedDRA_concept_id',
                'concept_name': 'SNOMED_concept_name',
                'concept_code': 'SNOMED_concept_code',
                'concept_class_id': 'SNOMED_concept_class_id'
            })
)
m_to_s_r['MedDRA_concept_id'] = m_to_s_r['MedDRA_concept_id'].astype(int)
m_to_s_r = m_to_s_r.reindex(np.sort(m_to_s_r.columns), axis=1)
print(m_to_s_r.shape)
print(m_to_s_r.SNOMED_concept_class_id.value_counts())
print(m_to_s_r.head())

# In[ ]:

# Get unique MedDRA concept IDs mapped to SNOMED
r2s = m_to_s_r.MedDRA_concept_id.unique()

# In[ ]:

# Find PTs and calculate intersection ratios
pts = (standard_reactions_meddra_relationships
       .query('MedDRA_concept_class_id_1=="PT"')
       .MedDRA_concept_id_1
       .unique())
print(len(np.intersect1d(pts, r2s)) / len(pts))
print(len(np.intersect1d(pts, r2s)) / len(r2s))

df = (standard_reactions_meddra_relationships
      .query('MedDRA_concept_id_1 in @r2s'))

print(df.shape)

# Join PTs with SNOMED
joinedpt = (df
            .set_index('MedDRA_concept_id_1')
            .join(m_to_s_r
                  .query('MedDRA_concept_id in @pts')
                  .set_index('MedDRA_concept_id')
                 )
            .rename_axis('MedDRA_concept_id_1')
            .reset_index()
            .rename(columns={
                'SNOMED_concept_id': 'SNOMED_concept_id_1',
                'SNOMED_concept_code': 'SNOMED_concept_code_1',
                'SNOMED_concept_name': 'SNOMED_concept_name_1',
                'SNOMED_concept_class_id': 'SNOMED_concept_class_id_1',
            })
            .dropna()
           )
joinedpt = joinedpt.reindex(np.sort(joinedpt.columns), axis=1)
print(joinedpt.shape)
print(joinedpt.head())

# In[ ]:

# Find HLTs and calculate intersection ratios
hlts = (joinedpt
        .query('MedDRA_concept_class_id_2=="HLT"')
        .MedDRA_concept_id_2
        .unique())
print(len(np.intersect1d(hlts, r2s)) / len(hlts))
print(len(np.intersect1d(hlts, r2s)) / len(r2s))

df = (joinedpt.copy())

print(df.shape)
print(df.head())

# Join HLTs with SNOMED
joinedhlt = (df
             .set_index('MedDRA_concept_id_2')
             .join(m_to_s_r
                   .query('MedDRA_concept_id in @hlts')
                   .set_index('MedDRA_concept_id')
                  )
             .rename_axis('MedDRA_concept_id_2')
             .reset_index()
             .rename(columns={
                 'SNOMED_concept_id': 'SNOMED_concept_id_2',
                 'SNOMED_concept_code': 'SNOMED_concept_code_2',
                 'SNOMED_concept_name': 'SNOMED_concept_name_2',
                 'SNOMED_concept_class_id': 'SNOMED_concept_class_id_2',
             })
            )
joinedhlt = joinedhlt.reindex(np.sort(joinedhlt.columns), axis=1)
print(joinedhlt.shape)
print(joinedhlt.head())

# In[ ]:

# Find HLGTS and calculate intersection ratios
hlgts = (joinedhlt
         .query('MedDRA_concept_class_id_3=="HLGT"')
         .MedDRA_concept_id_3
         .unique())
print(len(np.intersect1d(hlgts, r2s)) / len(hlgts))
print(len(np.intersect1d(hlgts, r2s)) / len(r2s))

df = (joinedhlt.copy())

print(df.shape)

# Join HLGTS with SNOMED
joinedhlgt = (df
              .set_index('MedDRA_concept_id_3')
              .join(m_to_s_r
                    .query('MedDRA_concept_id in @hlgts')
                    .set_index('MedDRA_concept_id')
                   )
              .rename_axis('MedDRA_concept_id_3')
              .reset_index()
              .drop_duplicates()
              .rename(columns={
                  'SNOMED_concept_id': 'SNOMED_concept_id_3',
                  'SNOMED_concept_code': 'SNOMED_concept_code_3',
                  'SNOMED_concept_name': 'SNOMED_concept_name_3',
                  'SNOMED_concept_class_id': 'SNOMED_concept_class_id_3',
              })
             )
joinedhlgt = joinedhlgt.reindex(np.sort(joinedhlgt.columns), axis=1)
print(joinedhlgt.shape)
print(joinedhlgt.head())

# In[ ]:

# Find SOCs and calculate intersection ratios
socs = (joinedhlgt
        .query('MedDRA_concept_class_id_4=="SOC"')
        .MedDRA_concept_id_4
        .unique())
print(len(np.intersect1d(socs, r2s)) / len(socs))
print(len(np.intersect1d(socs, r2s)) / len(r2s))

df = (joinedhlgt.copy())

print(df.shape)
print(df.head())
print(m_to_s_r.shape)
print(m_to_s_r.head())

# Join SOCs with SNOMED
joinedsoc = (df
             .set_index('MedDRA_concept_id_4')
             .join(m_to_s_r
                   .query('MedDRA_concept_id in @socs')
                   .set_index('MedDRA_concept_id')
                  )
             .rename_axis('MedDRA_concept_id_4')
             .reset_index()
             .drop_duplicates()
             .rename(columns={
                 'SNOMED_concept_id': 'SNOMED_concept_id_4',
                 'SNOMED_concept_code': 'SNOMED_concept_code_4',
                 'SNOMED_concept_name': 'SNOMED_concept_name_4',
                 'SNOMED_concept_class_id': 'SNOMED_concept_class_id_4',
             })
            )
joinedsoc = joinedsoc.reindex(np.sort(joinedsoc.columns), axis=1)
print(joinedsoc.shape)
print(joinedsoc.head())

# In[ ]:

# Calculate intersection ratios for PT concepts
smeddraconcepts = joinedpt.MedDRA_concept_id_1.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships
                     .query('MedDRA_concept_class_id_1=="PT"')
                     .MedDRA_concept_id_1
                     .unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(allmeddraconcepts))

# In[ ]:

# Calculate intersection ratios for HLT concepts
smeddraconcepts = joinedhlt.MedDRA_concept_id_2.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships
                     .query('MedDRA_concept_class_id_2=="HLT"')
                     .MedDRA_concept_id_2
                     .unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(allmeddraconcepts))

# In[ ]:

# Calculate intersection ratios for HLGT concepts
smeddraconcepts = joinedhlgt.MedDRA_concept_id_3.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships
                     .query('MedDRA_concept_class_id_3=="HLGT"')
                     .MedDRA_concept_id_3
                     .unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(allmeddraconcepts))

# In[ ]:

# Calculate intersection ratios for SOC concepts
smeddraconcepts = joinedsoc.MedDRA_concept_id_4.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships
                     .query('MedDRA_concept_class_id_4=="SOC"')
                     .MedDRA_concept_id_4
                     .unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts, allmeddraconcepts)) / len(allmeddraconcepts))

# In[ ]:

# Display joinedsoc data
print(joinedsoc.head())
print(joinedsoc.shape)
print(joinedsoc[joinedsoc.SNOMED_concept_id_1.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_1.nunique())
print(joinedsoc[joinedsoc.SNOMED_concept_id_2.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_2.nunique())
print(joinedsoc[joinedsoc.SNOMED_concept_id_3.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_3.nunique())
print(joinedsoc[joinedsoc.SNOMED_concept_id_4.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_4.nunique())

# In[ ]:

# Convert SNOMED concept codes to appropriate types
joinedsoc.SNOMED_concept_code_1 = joinedsoc.SNOMED_concept_code_1.astype(int)
joinedsoc.SNOMED_concept_code_2 = joinedsoc.SNOMED_concept_code_2.astype(float)
joinedsoc.SNOMED_concept_code_3 = joinedsoc.SNOMED_concept_code_3.astype(float)
joinedsoc.SNOMED_concept_code_4 = joinedsoc.SNOMED_concept_code_4.astype(float)

# In[ ]:

# Read standard reactions data
standard_reactions = pd.read_csv(er_dir + 'standard_reactions.csv.gz',
                                 compression="gzip",
                                 dtype={'safetyreportid': 'str'})
all_reports = standard_reactions.safetyreportid.unique()
print(standard_reactions.shape)
print(standard_reactions.head())

# In[ ]:

# Join MedDRA PT to SNOMED
standard_reactions_meddrapt_to_snomed = (joinedsoc
                                         .loc[:, ['MedDRA_concept_id_1', 'SNOMED_concept_id_1',
                                                  'SNOMED_concept_code_1', 'SNOMED_concept_name_1',
                                                  'SNOMED_concept_class_id_1']]
                                         .drop_duplicates()
                                         .rename(columns={
                                             'SNOMED_concept_id_1': 'SNOMED_concept_id',
                                             'SNOMED_concept_code_1': 'SNOMED_concept_code',
                                             'SNOMED_concept_name_1': 'SNOMED_concept_name',
                                             'SNOMED_concept_class_id_1': 'SNOMED_concept_class_id'
                                         })
                                         .set_index('MedDRA_concept_id_1')
                                         .join(standard_reactions
                                               .drop_duplicates()
                                               .set_index('MedDRA_concept_id')
                                              )
                                         .reset_index(drop=True)
                                         .drop(['MedDRA_concept_code', 'MedDRA_concept_name',
                                                'MedDRA_concept_class_id'], axis=1)
                                         .dropna()
)
standard_reactions_meddrapt_to_snomed = standard_reactions_meddrapt_to_snomed.reindex(np.sort(standard_reactions_meddrapt_to_snomed.columns), axis=1)
print(standard_reactions_meddrapt_to_snomed.shape)
print(standard_reactions_meddrapt_to_snomed.head())

# In[ ]:

# Calculate intersection ratio for SNOMED
intersection_ratio_snomed = len(
    np.intersect1d(
        all_reports,
        standard_reactions_meddrapt_to_snomed.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_snomed)

# In[ ]:

# Save SNOMED data to CSV
standard_reactions_meddrapt_to_snomed.to_csv(er_dir + 'standard_reactions_snomed.csv.gz', compression='gzip', index=False)

# In[ ]:

# Join MedDRA HLT to SNOMED
standard_reactions_meddrahlt_to_snomed = (joinedsoc
                                          .query('MedDRA_concept_class_id_2=="HLT"')
                                          .loc[:, ['MedDRA_concept_id_1', 'SNOMED_concept_id_2',
                                                   'SNOMED_concept_code_2', 'SNOMED_concept_name_2',
                                                   'SNOMED_concept_class_id_2']]
                                          .drop_duplicates()
                                          .rename(columns={
                                              'SNOMED_concept_id_2': 'SNOMED_concept_id',
                                              'SNOMED_concept_code_2': 'SNOMED_concept_code',
                                              'SNOMED_concept_name_2': 'SNOMED_concept_name',
                                              'SNOMED_concept_class_id_2': 'SNOMED_concept_class_id'
                                          })
                                          .set_index('MedDRA_concept_id_1')
                                          .join(standard_reactions
                                                .drop_duplicates()
                                                .set_index('MedDRA_concept_id')
                                               )
                                          .rename_axis('MedDRA_concept_id')
                                          .reset_index()
                                          .dropna(subset=['MedDRA_concept_id', 'SNOMED_concept_id', 'safetyreportid'])
)
standard_reactions_meddrahlt_to_snomed = standard_reactions_meddrahlt_to_snomed.reindex(np.sort(standard_reactions_meddrahlt_to_snomed.columns), axis=1)
print(standard_reactions_meddrahlt_to_snomed.shape)
print(standard_reactions_meddrahlt_to_snomed.head())

# In[ ]:

# Calculate intersection ratio for HLT to SNOMED
intersection_ratio_hlt_snomed = len(
    np.intersect1d(
        all_reports,
        standard_reactions_meddrahlt_to_snomed.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_hlt_snomed)

# In[ ]:

# Join MedDRA HLGT to SNOMED
standard_reactions_meddrahlgt_to_snomed = (joinedsoc
                                           .query('MedDRA_concept_class_id_2=="HLGT"')
                                           .loc[:, ['MedDRA_concept_id_1', 'SNOMED_concept_id_3',
                                                    'SNOMED_concept_code_3', 'SNOMED_concept_name_3',
                                                    'SNOMED_concept_class_id_2']]
                                           .drop_duplicates()
                                           .rename(columns={
                                               'SNOMED_concept_id_3': 'SNOMED_concept_id',
                                               'SNOMED_concept_code_3': 'SNOMED_concept_code',
                                               'SNOMED_concept_name_3': 'SNOMED_concept_name',
                                               'SNOMED_concept_class_id_3': 'SNOMED_concept_class_id'
                                           })
                                           .set_index('MedDRA_concept_id_1')
                                           .join(standard_reactions
                                                 .drop_duplicates()
                                                 .set_index('MedDRA_concept_id')
                                                )
                                           .rename_axis('MedDRA_concept_id')
                                           .reset_index()
                                           .dropna(subset=['MedDRA_concept_id', 'SNOMED_concept_id', 'safetyreportid'])
)
standard_reactions_meddrahlgt_to_snomed = standard_reactions_meddrahlgt_to_snomed.reindex(np.sort(standard_reactions_meddrahlgt_to_snomed.columns), axis=1)
print(standard_reactions_meddrahlgt_to_snomed.shape)
print(standard_reactions_meddrahlgt_to_snomed.head())

# In[ ]:

# Calculate intersection ratio for HLGT to SNOMED
intersection_ratio_hlgt_snomed = len(
    np.intersect1d(
        all_reports,
        standard_reactions_meddrahlgt_to_snomed.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_hlgt_snomed)

# In[ ]:

# Join MedDRA SOC to SNOMED
standard_reactions_meddrasoc_to_snomed = (joinedsoc
                                          .query('MedDRA_concept_class_id_4=="SOC"')
                                          .loc[:, ['MedDRA_concept_id_1', 'SNOMED_concept_id_4',
                                                   'SNOMED_concept_code_4', 'SNOMED_concept_name_4',
                                                   'SNOMED_concept_class_id_4']]
                                          .drop_duplicates()
                                          .rename(columns={
                                              'SNOMED_concept_id_4': 'SNOMED_concept_id',
                                              'SNOMED_concept_code_4': 'SNOMED_concept_code',
                                              'SNOMED_concept_name_4': 'SNOMED_concept_name',
                                              'SNOMED_concept_class_id_4': 'SNOMED_concept_class_id'
                                          })
                                          .set_index('MedDRA_concept_id_1')
                                          .join(standard_reactions
                                                .drop_duplicates()
                                                .set_index('MedDRA_concept_id')
                                               )
                                          .rename_axis('MedDRA_concept_id')
                                          .reset_index()
                                          .dropna(subset=['MedDRA_concept_id', 'SNOMED_concept_id', 'safetyreportid'])
)
standard_reactions_meddrasoc_to_snomed = standard_reactions_meddrasoc_to_snomed.reindex(np.sort(standard_reactions_meddrasoc_to_snomed.columns), axis=1)
print(standard_reactions_meddrasoc_to_snomed.shape)
print(standard_reactions_meddrasoc_to_snomed.head())

# In[ ]:

# Calculate intersection ratio for SOC to SNOMED
intersection_ratio_soc_snomed = len(
    np.intersect1d(
        all_reports,
        standard_reactions_meddrasoc_to_snomed.safetyreportid.astype(str).unique()
    )
) / len(all_reports)
print(intersection_ratio_soc_snomed)

# In[ ]:

# Clean up memory by deleting unused variables
del m_to_s_r
del df
del joinedpt
del joinedhlt
del joinedhlgt
del joinedsoc
del all_reports
del standard_reactions
del standard_reactions_meddrapt_to_snomed
del standard_reactions_meddrahlt_to_snomed
del standard_reactions_meddrahlgt_to_snomed
del standard_reactions_meddra_relationships
