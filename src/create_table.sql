-- Auto-generated Athena CREATE TABLE
-- Source: players_data.csv
-- Columns: 92
-- S3: s3://football-alpha-analysis-doruk/data/

DROP TABLE IF EXISTS football_analytics.players;

CREATE EXTERNAL TABLE football_analytics.players (
  `rk` double,
  `player` string,
  `nation` string,
  `pos` string,
  `squad` string,
  `comp` string,
  `age` double,
  `born` double,
  `mp` double,
  `starts` double,
  `min` double,
  `col_90s` double,
  `gls` double,
  `ast` double,
  `g_a` double,
  `g_pk` double,
  `pk` double,
  `pkatt` double,
  `crdy` double,
  `crdr` double,
  `g_a_pk` double,
  `gls_stats_shooting` double,
  `sh` double,
  `sot` double,
  `sotpct` double,
  `sh_90` double,
  `sot_90` double,
  `g_sh` double,
  `g_sot` double,
  `pk_stats_shooting` double,
  `pkatt_stats_shooting` double,
  `mp_stats_keeper` double,
  `starts_stats_keeper` double,
  `min_stats_keeper` double,
  `ga` double,
  `ga90` double,
  `sota` double,
  `saves` double,
  `savepct` double,
  `w` double,
  `d` double,
  `l` double,
  `cs` double,
  `cspct` double,
  `pkatt_stats_keeper` double,
  `pka` double,
  `pksv` double,
  `pkm` double,
  `mp_stats_playing_time` double,
  `min_stats_playing_time` double,
  `mn_mp` double,
  `minpct` double,
  `starts_stats_playing_time` double,
  `mn_start` double,
  `compl` double,
  `subs` double,
  `mn_sub` double,
  `unsub` double,
  `ppm` double,
  `ong` double,
  `onga` double,
  `empty_col` double,
  `col_90` double,
  `on_off` double,
  `crdy_stats_misc` double,
  `crdr_stats_misc` double,
  `col_2crdy` double,
  `fls` double,
  `fld` double,
  `off` double,
  `crs` double,
  `int` double,
  `tklw` double,
  `pkwon` double,
  `pkcon` double,
  `og` double,
  `us_shots` double,
  `us_key_passes` double,
  `us_npg` double,
  `col_90s_2` double,
  `finishing_alpha` double,
  `xag` double,
  `playmaking_alpha` double,
  `xg` double,
  `npxg` double,
  `xgchain` double,
  `xgbuildup` double,
  `gls_per90` double,
  `xg_per90` double,
  `ast_per90` double,
  `xag_per90` double,
  `alpha_per90` double
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://football-alpha-analysis-doruk/data/'
TBLPROPERTIES (
  'skip.header.line.count'='1',
  'serialization.null.format'='',
  'use.null.for.invalid.data'='true'
);

-- Verification queries
SELECT COUNT(*) AS total_players FROM football_analytics.players;

SELECT player, squad, comp, gls, xg, finishing_alpha
FROM football_analytics.players
WHERE CAST(gls AS DOUBLE) > 10
ORDER BY CAST(finishing_alpha AS DOUBLE) DESC
LIMIT 20;
