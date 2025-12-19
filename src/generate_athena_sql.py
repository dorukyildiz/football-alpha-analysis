import re

# CSV header
header = """Rk,Player,Nation,Pos,Squad,Comp,Age,Born,MP,Starts,Min,90s,Gls,Ast,G+A,G-PK,PK,PKatt,CrdY,CrdR,xG,npxG,xAG,npxG+xAG,PrgC,PrgP,PrgR,G+A-PK,xG+xAG,Rk_stats_shooting,Nation_stats_shooting,Pos_stats_shooting,Comp_stats_shooting,Age_stats_shooting,Born_stats_shooting,90s_stats_shooting,Gls_stats_shooting,Sh,SoT,SoT%,Sh/90,SoT/90,G/Sh,G/SoT,Dist,FK,PK_stats_shooting,PKatt_stats_shooting,xG_stats_shooting,npxG_stats_shooting,npxG/Sh,G-xG,np:G-xG,Rk_stats_passing,Nation_stats_passing,Pos_stats_passing,Comp_stats_passing,Age_stats_passing,Born_stats_passing,90s_stats_passing,Cmp,Att,Cmp%,TotDist,PrgDist,Ast_stats_passing,xAG_stats_passing,xA,A-xAG,KP,1/3,PPA,CrsPA,PrgP_stats_passing,Rk_stats_passing_types,Nation_stats_passing_types,Pos_stats_passing_types,Comp_stats_passing_types,Age_stats_passing_types,Born_stats_passing_types,90s_stats_passing_types,Att_stats_passing_types,Live,Dead,FK_stats_passing_types,TB,Sw,Crs,TI,CK,In,Out,Str,Cmp_stats_passing_types,Off,Blocks,Rk_stats_gca,Nation_stats_gca,Pos_stats_gca,Comp_stats_gca,Age_stats_gca,Born_stats_gca,90s_stats_gca,SCA,SCA90,PassLive,PassDead,TO,Sh_stats_gca,Fld,Def,GCA,GCA90,Rk_stats_defense,Nation_stats_defense,Pos_stats_defense,Comp_stats_defense,Age_stats_defense,Born_stats_defense,90s_stats_defense,Tkl,TklW,Def 3rd,Mid 3rd,Att 3rd,Att_stats_defense,Tkl%,Lost,Blocks_stats_defense,Sh_stats_defense,Pass,Int,Tkl+Int,Clr,Err,Rk_stats_possession,Nation_stats_possession,Pos_stats_possession,Comp_stats_possession,Age_stats_possession,Born_stats_possession,90s_stats_possession,Touches,Def Pen,Def 3rd_stats_possession,Mid 3rd_stats_possession,Att 3rd_stats_possession,Att Pen,Live_stats_possession,Att_stats_possession,Succ,Succ%,Tkld,Tkld%,Carries,TotDist_stats_possession,PrgDist_stats_possession,PrgC_stats_possession,1/3_stats_possession,CPA,Mis,Dis,Rec,PrgR_stats_possession,Rk_stats_playing_time,Nation_stats_playing_time,Pos_stats_playing_time,Comp_stats_playing_time,Age_stats_playing_time,Born_stats_playing_time,MP_stats_playing_time,Min_stats_playing_time,Mn/MP,Min%,90s_stats_playing_time,Starts_stats_playing_time,Mn/Start,Compl,Subs,Mn/Sub,unSub,PPM,onG,onGA,+/-,+/-90,On-Off,onxG,onxGA,xG+/-,xG+/-90,Rk_stats_misc,Nation_stats_misc,Pos_stats_misc,Comp_stats_misc,Age_stats_misc,Born_stats_misc,90s_stats_misc,CrdY_stats_misc,CrdR_stats_misc,2CrdY,Fls,Fld_stats_misc,Off_stats_misc,Crs_stats_misc,Int_stats_misc,TklW_stats_misc,PKwon,PKcon,OG,Recov,Won,Lost_stats_misc,Won%,Rk_stats_keeper,Nation_stats_keeper,Pos_stats_keeper,Comp_stats_keeper,Age_stats_keeper,Born_stats_keeper,MP_stats_keeper,Starts_stats_keeper,Min_stats_keeper,90s_stats_keeper,GA,GA90,SoTA,Saves,Save%,W,D,L,CS,CS%,PKatt_stats_keeper,PKA,PKsv,PKm,Rk_stats_keeper_adv,Nation_stats_keeper_adv,Pos_stats_keeper_adv,Comp_stats_keeper_adv,Age_stats_keeper_adv,Born_stats_keeper_adv,90s_stats_keeper_adv,GA_stats_keeper_adv,PKA_stats_keeper_adv,FK_stats_keeper_adv,CK_stats_keeper_adv,OG_stats_keeper_adv,PSxG,PSxG/SoT,PSxG+/-,/90,Cmp_stats_keeper_adv,Att_stats_keeper_adv,Cmp%_stats_keeper_adv,Att (GK),Thr,Launch%,AvgLen,Opp,Stp,Stp%,#OPA,#OPA/90,AvgDist"""

columns = header.split(',')


def clean_column_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if not name:
        return 'empty_col'
    if name[0].isdigit():
        name = 'col_' + name
    return name


# String columns
string_cols = {'player', 'nation', 'pos', 'squad', 'comp',
               'nation_stats_shooting', 'pos_stats_shooting', 'comp_stats_shooting',
               'nation_stats_passing', 'pos_stats_passing', 'comp_stats_passing',
               'nation_stats_passing_types', 'pos_stats_passing_types', 'comp_stats_passing_types',
               'nation_stats_gca', 'pos_stats_gca', 'comp_stats_gca',
               'nation_stats_defense', 'pos_stats_defense', 'comp_stats_defense',
               'nation_stats_possession', 'pos_stats_possession', 'comp_stats_possession',
               'nation_stats_playing_time', 'pos_stats_playing_time', 'comp_stats_playing_time',
               'nation_stats_misc', 'pos_stats_misc', 'comp_stats_misc',
               'nation_stats_keeper', 'pos_stats_keeper', 'comp_stats_keeper',
               'nation_stats_keeper_adv', 'pos_stats_keeper_adv', 'comp_stats_keeper_adv',
               'age_stats_shooting', 'age_stats_passing', 'age_stats_passing_types',
               'age_stats_gca', 'age_stats_defense', 'age_stats_possession',
               'age_stats_playing_time', 'age_stats_misc', 'age_stats_keeper', 'age_stats_keeper_adv'}

# Track used names to avoid duplicates
used_names = {}
sql_columns = []

for col in columns:
    clean_name = clean_column_name(col)

    # Handle duplicates
    if clean_name in used_names:
        used_names[clean_name] += 1
        clean_name = f"{clean_name}_{used_names[clean_name]}"
    else:
        used_names[clean_name] = 1

    if clean_name.rstrip('_0123456789') in string_cols or clean_name in string_cols:
        dtype = 'string'
    else:
        dtype = 'double'
    sql_columns.append(f"  `{clean_name}` {dtype}")

sql = f"""CREATE EXTERNAL TABLE football_analytics.players (
{chr(44).join(sql_columns)}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\\\'
)
STORED AS TEXTFILE
LOCATION 's3://football-alpha-analysis-doruk/data/'
TBLPROPERTIES (
  'skip.header.line.count'='1',
  'serialization.null.format'='',
  'use.null.for.invalid.data'='true'
);"""

print(sql)

with open('create_table.sql', 'w') as f:
    f.write(sql)
print("\n\nSQL saved to create_table.sql")