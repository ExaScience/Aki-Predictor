import psycopg2
from sqlalchemy import create_engine
import pandas as pd

def test_postgres(cursor):

    cursor.execute("""
SELECT
    *
FROM
   pg_catalog.pg_tables
WHERE
   schemaname != 'pg_catalog'
AND schemaname != 'information_schema';
    """)

    tables = cursor.fetchall()

    print("List of tables:")
    for table in tables:
        print(table)

    # construct an engine connection string
    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user = "",
        password = "",
        host = "",
        port = "",
        database = mimicDB,
    )

    cmd = 'psql "postgresql+psycopg2://postgres:postgres@localhost:5432/mimic" '

    # create sqlalchemy engine
    engine = create_engine(engine_string)

    # read a table from database into pandas dataframe
    for table in tables:
        tn = table[1]
        print("Table {}".format(tn))

        df = pd.read_sql_table(tn, engine)

        print("Head of extracted pandas DF:")
        print(df.head())

def urine_output(cursor):
        
    view ="DROP MATERIALIZED VIEW IF EXISTS urineoutput CASCADE; \
            CREATE MATERIALIZED VIEW urineoutput as \
            select oe.icustay_id, oe.charttime \
            , SUM( \
                case when oe.itemid = 227488 then -1*value \
                else value end \
              ) as value \
            from outputevents oe \
            where oe.itemid in \
            ( \
              40055, \
              43175, \
              40069, \
              40094, \
              40715, \
              40473, \
              40085, \
              40057, \
              40056, \
              40405, \
              40428, \
              40086, \
              40096, \
              40651, \
              226559, \
              226560, \
              226561, \
              226584, \
              226563, \
              226564, \
              226565, \
              226567, \
              226557, \
              226558, \
              227488, \
              227489  \
            ) \
            and oe.value < 5000 \
            and oe.icustay_id is not null \
            group by icustay_id, charttime;"
            
    cursor.execute(view)

def creatinine(cursor):
    
    view = "DROP MATERIALIZED VIEW IF EXISTS kdigo_creat CASCADE; \
            CREATE MATERIALIZED VIEW kdigo_creat as \
            with cr as \
            ( \
            select \
                ie.icustay_id \
              , ie.intime, ie.outtime \
              , le.valuenum as creat \
              , le.charttime \
              from icustays ie \
              left join labevents le \
                on ie.subject_id = le.subject_id \
                and le.ITEMID = 50912 \
                and le.VALUENUM is not null \
                and le.CHARTTIME between (ie.intime - interval '7' day) and (ie.intime + interval '7' day) \
            ) \
            SELECT \
              cr.icustay_id \
              , cr.charttime \
              , cr.creat \
              , MIN(cr48.creat) AS creat_low_past_48hr \
              , MIN(cr7.creat) AS creat_low_past_7day \
            FROM cr \
            LEFT JOIN cr cr48 \
              ON cr.icustay_id = cr48.icustay_id \
              AND cr48.charttime <  cr.charttime \
              AND cr48.charttime >= (cr.charttime - INTERVAL '48' HOUR) \
            LEFT JOIN cr cr7 \
              ON cr.icustay_id = cr7.icustay_id \
              AND cr7.charttime <  cr.charttime \
              AND cr7.charttime >= (cr.charttime - INTERVAL '7' DAY) \
            GROUP BY cr.icustay_id, cr.charttime, cr.creat \
            ORDER BY cr.icustay_id, cr.charttime, cr.creat;"
    
    cursor.execute(view)        

def weight_duration(cursor):
    
    #-- This query extracts weights for ICU patients with start/stop times 
    #-- if only an admission weight is given, then this is assigned from intime to outtime 
            
    view = " DROP MATERIALIZED VIEW IF EXISTS weightdurations CASCADE; \
            CREATE MATERIALIZED VIEW weightdurations as \
            WITH wt_neonate AS \
            (\
                SELECT c.icustay_id, c.charttime \
                , MAX(CASE WHEN c.itemid = 3580 THEN c.valuenum END) as wt_kg \
                , MAX(CASE WHEN c.itemid = 3581 THEN c.valuenum END) as wt_lb \
                , MAX(CASE WHEN c.itemid = 3582 THEN c.valuenum END) as wt_oz \
                FROM chartevents c \
                WHERE c.itemid in (3580, 3581, 3582) \
                AND c.icustay_id IS NOT NULL \
                AND c.error IS DISTINCT FROM 1 \
                AND c.valuenum > 0 \
                GROUP BY c.icustay_id, c.charttime \
            ) \
            , birth_wt AS \
            ( \
                SELECT c.icustay_id, c.charttime \
                , MAX( \
                  CASE \
                  WHEN c.itemid = 4183 THEN \
                    CASE \
                      WHEN c.value ~ '[^0-9\.]' THEN NULL \
                      WHEN CAST(c.value AS NUMERIC) > 100 THEN CAST(c.value AS NUMERIC)/1000 \
                      WHEN CAST(c.value AS NUMERIC) < 10 THEN CAST(c.value AS NUMERIC) \
                    ELSE NULL END \
                  WHEN c.itemid = 3723 AND c.valuenum < 10 THEN c.valuenum \
                  ELSE NULL END) as wt_kg \
                FROM chartevents c \
                WHERE c.itemid in (3723, 4183) \
                AND c.icustay_id IS NOT NULL \
                AND c.error IS DISTINCT FROM 1 \
                GROUP BY c.icustay_id, c.charttime \
            ) \
            , wt_stg as \
            ( \
                SELECT \
                    c.icustay_id \
                  , c.charttime \
                  , case when c.itemid in (762,226512) then 'admit' \
                    else 'daily' end as weight_type \
                  , c.valuenum as weight \
                FROM chartevents c \
                WHERE c.valuenum IS NOT NULL \
                  AND c.itemid in \
                  ( \
                      762,226512 \
                    , 763,224639 \
                  ) \
                  AND c.icustay_id IS NOT NULL \
                  AND c.valuenum > 0 \
                  AND c.error IS DISTINCT FROM 1 \
                UNION ALL \
                SELECT \
                    n.icustay_id \
                  , n.charttime \
                  , 'daily' AS weight_type \
                  , CASE \
                      WHEN wt_kg IS NOT NULL THEN wt_kg \
                      WHEN wt_lb IS NOT NULL THEN wt_lb*0.45359237 + wt_oz*0.0283495231 \
                    ELSE NULL END AS weight \
                FROM wt_neonate n \
                UNION ALL \
                SELECT \
                    b.icustay_id \
                  , b.charttime \
                  , 'admit' AS weight_type \
                  , wt_kg as weight \
                FROM birth_wt b \
            ) \
            , wt_stg1 as \
            ( \
              select \
                  icustay_id \
                , charttime \
                , weight_type \
                , weight \
                , ROW_NUMBER() OVER (partition by icustay_id, weight_type order by charttime) as rn \
              from wt_stg \
              WHERE weight IS NOT NULL \
            ) \
           , wt_stg2 AS \
            ( \
              SELECT \
                  wt_stg1.icustay_id \
                , ie.intime, ie.outtime \
                , case when wt_stg1.weight_type = 'admit' and wt_stg1.rn = 1 \
                    then ie.intime - interval '2' hour \
                  else wt_stg1.charttime end as starttime \
                , wt_stg1.weight \
              from wt_stg1 \
              INNER JOIN icustays ie \
                on ie.icustay_id = wt_stg1.icustay_id \
            ) \
            , wt_stg3 as \
            ( \
              select \
                icustay_id \
                , intime, outtime \
                , starttime \
                , coalesce( \
                    LEAD(starttime) OVER (PARTITION BY icustay_id ORDER BY starttime), \
                    outtime + interval '2' hour \
                  ) as endtime \
                , weight \
              from wt_stg2 \
            ) \
           , wt1 as \
            ( \
              select \
                  icustay_id \
                , starttime \
                , coalesce(endtime, \
                  LEAD(starttime) OVER (partition by icustay_id order by starttime), \
                 outtime + interval '2' hour) \
                as endtime \
                , weight \
              from wt_stg3 \
            ) \
            , wt_fix as \
            ( \
              select ie.icustay_id \
                , ie.intime - interval '2' hour as starttime \
                , wt.starttime as endtime \
                , wt.weight \
              from icustays ie \
              inner join \
              ( \
                SELECT wt1.icustay_id, wt1.starttime, wt1.weight \
                , ROW_NUMBER() OVER (PARTITION BY wt1.icustay_id ORDER BY wt1.starttime) as rn \
                FROM wt1 \
              ) wt \
                ON  ie.icustay_id = wt.icustay_id \
                AND wt.rn = 1 \
                and ie.intime < wt.starttime \
            ) \
            , wt2 as \
            ( \
              select \
                  wt1.icustay_id \
                , wt1.starttime \
                , wt1.endtime \
                , wt1.weight \
              from wt1 \
              UNION \
              SELECT \
                  wt_fix.icustay_id \
                , wt_fix.starttime \
                , wt_fix.endtime \
                , wt_fix.weight \
              from wt_fix \
            ) \
            , echo_lag as \
            ( \
              select \
                ie.icustay_id \
                , ie.intime, ie.outtime \
                , 0.453592*ec.weight as weight_echo \
                , ROW_NUMBER() OVER (PARTITION BY ie.icustay_id ORDER BY ec.charttime) as rn \
                , ec.charttime as starttime \
                , LEAD(ec.charttime) OVER (PARTITION BY ie.icustay_id ORDER BY ec.charttime) as endtime \
              from icustays ie \
              inner join echodata ec \
                on ie.hadm_id = ec.hadm_id \
              where ec.weight is not null \
            ) \
            , echo_final as \
            ( \
                select \
                  el.icustay_id \
                  , el.starttime \
                  , coalesce(el.endtime, el.outtime + interval '2' hour) as endtime \
                  , weight_echo \
                from echo_lag el \
                UNION \
                select \
                  el.icustay_id \
                  , el.intime - interval '2' hour as starttime \
                  , el.starttime as endtime \
                  , el.weight_echo \
                from echo_lag el \
                where el.rn = 1 \
                and el.starttime > el.intime - interval '2' hour \
            ) \
            select \
              wt2.icustay_id, wt2.starttime, wt2.endtime, wt2.weight \
            from wt2 \
            UNION \
            select \
              ef.icustay_id, ef.starttime, ef.endtime, ef.weight_echo as weight \
            from echo_final ef \
            where ef.icustay_id not in (select distinct icustay_id from wt2) \
            order by icustay_id, starttime, endtime;"
               
    cursor.execute(view)

def urine_kidigo(cursor):
    
#      -- we have joined each row to all rows preceding within 24 hours \
#               -- we can now sum these rows to get total UO over the last 24 hours \
#               -- we can use case statements to restrict it to only the last 6/12 hours \
#               -- therefore we have three sums: \
#               -- 1) over a 6 hour period \
#               -- 2) over a 12 hour period \
#               -- 3) over a 24 hour period \
#               -- note that we assume data charted at charttime corresponds to 1 hour of UO \
#               -- therefore we use '5' and '11' to restrict the period, rather than 6/12 \
#               -- this assumption may overestimate UO rate when documentation is done less than hourly \
#               -- 6 hours \
              
    view= " DROP MATERIALIZED VIEW IF EXISTS kdigo_uo CASCADE; \
            CREATE MATERIALIZED VIEW kdigo_uo AS \
            with ur_stg as \
            ( \
              select io.icustay_id, io.charttime \
              , sum(case when io.charttime <= iosum.charttime + interval '5' hour \
                  then iosum.VALUE \
                else null end) as UrineOutput_6hr \
              , sum(case when io.charttime <= iosum.charttime + interval '11' hour \
                  then iosum.VALUE \
                else null end) as UrineOutput_12hr \
              , sum(iosum.VALUE) as UrineOutput_24hr \
              , ROUND(CAST(EXTRACT(EPOCH FROM \
                  io.charttime -  \
                    MIN(case when io.charttime <= iosum.charttime + interval '5' hour \
                      then iosum.charttime \
                    else null end) \
                )/3600.0 AS NUMERIC), 4) AS uo_tm_6hr \
              , ROUND(CAST(EXTRACT(EPOCH FROM \
                  io.charttime -  \
                    MIN(case when io.charttime <= iosum.charttime + interval '11' hour \
                      then iosum.charttime \
                    else null end) \
               )/3600.0 AS NUMERIC), 4) AS uo_tm_12hr \
              , ROUND(CAST(EXTRACT(EPOCH FROM \
                  io.charttime - MIN(iosum.charttime) \
               )/3600.0 AS NUMERIC), 4) AS uo_tm_24hr \
              from urineoutput io \
              left join urineoutput iosum \
                on  io.icustay_id = iosum.icustay_id \
                and io.charttime >= iosum.charttime \
                and io.charttime <= (iosum.charttime + interval '23' hour) \
              group by io.icustay_id, io.charttime \
            ) \
            select \
              ur.icustay_id \
            , ur.charttime \
            , wd.weight \
            , ur.UrineOutput_6hr \
            , ur.UrineOutput_12hr \
            , ur.UrineOutput_24hr \
            , ROUND((ur.UrineOutput_6hr/wd.weight/(uo_tm_6hr+1))::NUMERIC, 4) AS uo_rt_6hr \
            , ROUND((ur.UrineOutput_12hr/wd.weight/(uo_tm_12hr+1))::NUMERIC, 4) AS uo_rt_12hr \
            , ROUND((ur.UrineOutput_24hr/wd.weight/(uo_tm_24hr+1))::NUMERIC, 4) AS uo_rt_24hr \
            , uo_tm_6hr \
            , uo_tm_12hr \
            , uo_tm_24hr \
            from ur_stg ur \
            left join weightdurations wd \
              on  ur.icustay_id = wd.icustay_id \
              and ur.charttime >= wd.starttime \
              and ur.charttime <  wd.endtime \
            order by icustay_id, charttime; "
            
    cursor.execute(view)

def kidigo_7_days_creatinine(cursor): 
   
    #-- This query checks if the patient had AKI during the first 7 days of their ICU
    #-- stay according to the KDIGO guideline.
    #-- https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf

    view = "DROP MATERIALIZED VIEW IF EXISTS kdigo_7_days_creatinine; \
            CREATE MATERIALIZED VIEW kdigo_7_days_creatinine AS  \
            WITH cr_aki AS  \
            (  \
              SELECT  \
                k.icustay_id  \
                , k.charttime  \
                , k.creat  \
                , k.aki_stage_creat  \
                , ROW_NUMBER() OVER (PARTITION BY k.icustay_id ORDER BY k.aki_stage_creat DESC, k.creat DESC) AS rn  \
              FROM icustays ie  \
              INNER JOIN kdigo_stages_creatinine k  \
                ON ie.icustay_id = k.icustay_id  \
              WHERE k.charttime > (ie.intime - interval '6' hour)  \
              AND k.charttime <= (ie.intime + interval '7' day)  \
              AND k.aki_stage_creat IS NOT NULL  \
            )  \
            select  \
                ie.icustay_id  \
              , cr.charttime as charttime_creat  \
              , cr.creat  \
              , cr.aki_stage_creat  \
              , cr.aki_stage_creat AS aki_stage_7day  \
              , CASE WHEN (cr.aki_stage_creat > 0) THEN 1 ELSE 0 END AS aki_7day  \
            FROM icustays ie  \
            LEFT JOIN cr_aki cr  \
              ON ie.icustay_id = cr.icustay_id  \
              AND cr.rn = 1  \
            order by ie.icustay_id; "
            
    cursor.execute(view)

def kidigo_stages_creatinine(cursor):
    
    #-- This query checks if the patient had AKI according to KDIGO.
    #-- AKI is calculated every time a creatinine or urine output measurement occurs.
    #-- Baseline creatinine is defined as the lowest creatinine in the past 7 days.

    view = " DROP MATERIALIZED VIEW IF EXISTS kdigo_stages_creatinine CASCADE; \
            CREATE MATERIALIZED VIEW kdigo_stages_creatinine AS \
            with cr_stg AS \
            ( \
              SELECT \
                cr.icustay_id \
                , cr.charttime \
                , cr.creat \
                , case \
                    when cr.creat >= (cr.creat_low_past_7day*3.0) then 3 \
                    when cr.creat >= 4 \
                    and (cr.creat_low_past_48hr <= 3.7 OR cr.creat >= (1.5*cr.creat_low_past_7day)) \
                        then 3  \
                    when cr.creat >= (cr.creat_low_past_7day*2.0) then 2 \
                    when cr.creat >= (cr.creat_low_past_48hr+0.3) then 1 \
                    when cr.creat >= (cr.creat_low_past_7day*1.5) then 1 \
                else 0 end as aki_stage_creat \
              FROM kdigo_creat cr \
            ) \
          , tm_stg AS \
            ( \
                SELECT \
                  icustay_id, charttime \
                FROM cr_stg \
            ) \
            select \
                ie.icustay_id \
              , tm.charttime \
              , cr.creat \
              , cr.aki_stage_creat \
              , cr.aki_stage_creat AS aki_stage \
            FROM icustays ie \
            LEFT JOIN tm_stg tm \
              ON ie.icustay_id = tm.icustay_id \
            LEFT JOIN cr_stg cr \
              ON ie.icustay_id = cr.icustay_id \
            AND tm.charttime = cr.charttime \
            order by ie.icustay_id, tm.charttime; "
            
    cursor.execute(view)
    
def kidigo_7_days(cursor): 
   
    #-- This query checks if the patient had AKI during the first 7 days of their ICU
    #-- stay according to the KDIGO guideline.
    #-- https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf

    view = "DROP MATERIALIZED VIEW IF EXISTS kdigo_stages_7day; \
            CREATE MATERIALIZED VIEW kdigo_stages_7day AS  \
            WITH cr_aki AS  \
            (  \
              SELECT  \
                k.icustay_id  \
                , k.charttime  \
                , k.creat  \
                , k.aki_stage_creat  \
                , ROW_NUMBER() OVER (PARTITION BY k.icustay_id ORDER BY k.aki_stage_creat DESC, k.creat DESC) AS rn  \
              FROM icustays ie  \
              INNER JOIN kdigo_stages k  \
                ON ie.icustay_id = k.icustay_id  \
              WHERE k.charttime > (ie.intime - interval '6' hour)  \
              AND k.charttime <= (ie.intime + interval '7' day)  \
              AND k.aki_stage_creat IS NOT NULL  \
            )  \
            , uo_aki AS  \
            ( \
              SELECT  \
                k.icustay_id  \
                , k.charttime  \
                , k.uo_rt_6hr, k.uo_rt_12hr, k.uo_rt_24hr  \
                , k.aki_stage_uo  \
                , ROW_NUMBER() OVER   \
                (  \
                  PARTITION BY k.icustay_id  \
                  ORDER BY k.aki_stage_uo DESC, k.uo_rt_24hr DESC, k.uo_rt_12hr DESC, k.uo_rt_6hr DESC  \
                ) AS rn  \
              FROM icustays ie  \
              INNER JOIN kdigo_stages k  \
                ON ie.icustay_id = k.icustay_id  \
              WHERE k.charttime > (ie.intime - interval '6' hour)  \
              AND k.charttime <= (ie.intime + interval '7' day)  \
              AND k.aki_stage_uo IS NOT NULL  \
            )  \
            select  \
                ie.icustay_id  \
              , cr.charttime as charttime_creat  \
              , cr.creat  \
              , cr.aki_stage_creat  \
              , uo.charttime as charttime_uo  \
              , uo.uo_rt_6hr  \
              , uo.uo_rt_12hr  \
              , uo.uo_rt_24hr  \
              , uo.aki_stage_uo  \
              , GREATEST(cr.aki_stage_creat,uo.aki_stage_uo) AS aki_stage_7day  \
              , CASE WHEN GREATEST(cr.aki_stage_creat, uo.aki_stage_uo) > 0 THEN 1 ELSE 0 END AS aki_7day  \
            FROM icustays ie  \
            LEFT JOIN cr_aki cr  \
              ON ie.icustay_id = cr.icustay_id  \
              AND cr.rn = 1  \
            LEFT JOIN uo_aki uo  \
              ON ie.icustay_id = uo.icustay_id  \
              AND uo.rn = 1  \
            order by ie.icustay_id; "
            
    cursor.execute(view)

def kidigo_stages(cursor):
    
    #-- This query checks if the patient had AKI according to KDIGO.
    #-- AKI is calculated every time a creatinine or urine output measurement occurs.
    #-- Baseline creatinine is defined as the lowest creatinine in the past 7 days.

    view = " DROP MATERIALIZED VIEW IF EXISTS kdigo_stages CASCADE; \
            CREATE MATERIALIZED VIEW kdigo_stages AS \
            with cr_stg AS \
            ( \
              SELECT \
                cr.icustay_id \
                , cr.charttime \
                , cr.creat \
                , case \
                    when cr.creat >= (cr.creat_low_past_7day*3.0) then 3 \
                    when cr.creat >= 4 \
                    and (cr.creat_low_past_48hr <= 3.7 OR cr.creat >= (1.5*cr.creat_low_past_7day)) \
                        then 3  \
                    when cr.creat >= (cr.creat_low_past_7day*2.0) then 2 \
                    when cr.creat >= (cr.creat_low_past_48hr+0.3) then 1 \
                    when cr.creat >= (cr.creat_low_past_7day*1.5) then 1 \
                else 0 end as aki_stage_creat \
              FROM kdigo_creat cr \
            ) \
            , uo_stg as \
            ( \
              select \
                  uo.icustay_id \
                , uo.charttime \
                , uo.weight \
                , uo.uo_rt_6hr \
                , uo.uo_rt_12hr \
                , uo.uo_rt_24hr \
                , CASE \
                    WHEN uo.uo_rt_6hr IS NULL THEN NULL \
                    WHEN uo.charttime <= ie.intime + interval '6' hour THEN 0 \
                    WHEN uo.uo_tm_24hr >= 11 AND uo.uo_rt_24hr < 0.3 THEN 3 \
                    WHEN uo.uo_tm_12hr >= 5 AND uo.uo_rt_12hr = 0 THEN 3 \
                    WHEN uo.uo_tm_12hr >= 5 AND uo.uo_rt_12hr < 0.5 THEN 2 \
                    WHEN uo.uo_tm_6hr >= 2 AND uo.uo_rt_6hr  < 0.5 THEN 1 \
                ELSE 0 END AS aki_stage_uo \
              from kdigo_uo uo \
              INNER JOIN icustays ie \
                ON uo.icustay_id = ie.icustay_id \
            ) \
            , tm_stg AS \
            ( \
                SELECT \
                  icustay_id, charttime \
                FROM cr_stg \
                UNION \
                SELECT \
                  icustay_id, charttime \
                FROM uo_stg \
            ) \
            select \
                ie.icustay_id \
              , tm.charttime \
              , cr.creat \
              , cr.aki_stage_creat \
              , uo.uo_rt_6hr \
              , uo.uo_rt_12hr \
              , uo.uo_rt_24hr \
              , uo.aki_stage_uo \
              , GREATEST(cr.aki_stage_creat, uo.aki_stage_uo) AS aki_stage \
            FROM icustays ie \
            LEFT JOIN tm_stg tm \
              ON ie.icustay_id = tm.icustay_id \
            LEFT JOIN cr_stg cr \
              ON ie.icustay_id = cr.icustay_id \
            AND tm.charttime = cr.charttime \
            LEFT JOIN uo_stg uo \
              ON ie.icustay_id = uo.icustay_id \
              AND tm.charttime = uo.charttime \
            order by ie.icustay_id, tm.charttime; "
            
    cursor.execute(view)

def get_labevents(cursor):

    #-- This query pivots lab values taken during the 7 first days of  a patient's stay
    #-- Have already confirmed that the unit of measurement is always the same: null or the correct unit

    #-- Extract all bicarbonate, blood urea nitrogen (BUN), calcium, chloride, creatinine, 
    #hemoglobin, international normalized ratio (INR), platelet, potassium, prothrombin time (PT), 
    #partial throm- boplastin time (PTT), and white blood count (WBC) values from labevents around patient's ICU stay
    
    view = "DROP MATERIALIZED VIEW IF EXISTS labstay CASCADE; \
            CREATE materialized VIEW labstay AS \
            SELECT \
                pvt.subject_id, pvt.hadm_id, pvt.icustay_id \
                  , min(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE null END) as ANIONGAP_min \
                  , max(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE null END) as ANIONGAP_max  \
                  , min(CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE null END) as ALBUMIN_min \
                  , max(CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE null END) as ALBUMIN_max \
                  , min(CASE WHEN label = 'BANDS' THEN valuenum ELSE null END) as BANDS_min \
                  , max(CASE WHEN label = 'BANDS' THEN valuenum ELSE null END) as BANDS_max \
                  , min(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE null END) as BICARBONATE_min \
                  , max(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE null END) as BICARBONATE_max \
                  , min(CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE null END) as BILIRUBIN_min \
                  , max(CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE null END) as BILIRUBIN_max \
                  , min(CASE WHEN label = 'CREATININE' THEN valuenum ELSE null END) as CREATININE_min \
                  , max(CASE WHEN label = 'CREATININE' THEN valuenum ELSE null END) as CREATININE_max \
                  , min(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE null END) as CHLORIDE_min \
                  , max(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE null END) as CHLORIDE_max \
                  , min(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE null END) as GLUCOSE_min \
                  , max(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE null END) as GLUCOSE_max \
                  , min(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE null END) as HEMATOCRIT_min \
                  , max(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE null END) as HEMATOCRIT_max \
                  , min(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE null END) as HEMOGLOBIN_min \
                  , max(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE null END) as HEMOGLOBIN_max \
                  , min(CASE WHEN label = 'LACTATE' THEN valuenum ELSE null END) as LACTATE_min \
                  , max(CASE WHEN label = 'LACTATE' THEN valuenum ELSE null END) as LACTATE_max \
                  , min(CASE WHEN label = 'PLATELET' THEN valuenum ELSE null END) as PLATELET_min \
                  , max(CASE WHEN label = 'PLATELET' THEN valuenum ELSE null END) as PLATELET_max \
                  , min(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE null END) as POTASSIUM_min \
                  , max(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE null END) as POTASSIUM_max \
                  , min(CASE WHEN label = 'PTT' THEN valuenum ELSE null END) as PTT_min \
                  , max(CASE WHEN label = 'PTT' THEN valuenum ELSE null END) as PTT_max \
                  , min(CASE WHEN label = 'INR' THEN valuenum ELSE null END) as INR_min \
                  , max(CASE WHEN label = 'INR' THEN valuenum ELSE null END) as INR_max \
                  , min(CASE WHEN label = 'PT' THEN valuenum ELSE null END) as PT_min \
                  , max(CASE WHEN label = 'PT' THEN valuenum ELSE null END) as PT_max \
                  , min(CASE WHEN label = 'SODIUM' THEN valuenum ELSE null END) as SODIUM_min \
                  , max(CASE WHEN label = 'SODIUM' THEN valuenum ELSE null end) as SODIUM_max \
                  , min(CASE WHEN label = 'BUN' THEN valuenum ELSE null end) as BUN_min \
                  , max(CASE WHEN label = 'BUN' THEN valuenum ELSE null end) as BUN_max \
                  , min(CASE WHEN label = 'WBC' THEN valuenum ELSE null end) as WBC_min \
                  , max(CASE WHEN label = 'WBC' THEN valuenum ELSE null end) as WBC_max \
            FROM \
            ( SELECT ie.subject_id, ie.hadm_id, ie.icustay_id \
              , CASE \
                    WHEN itemid = 50868 THEN 'ANION GAP' \
                    WHEN itemid = 50862 THEN 'ALBUMIN' \
                    WHEN itemid = 51144 THEN 'BANDS' \
                    WHEN itemid = 50882 THEN 'BICARBONATE' \
                    WHEN itemid = 50885 THEN 'BILIRUBIN' \
                    WHEN itemid = 50912 THEN 'CREATININE' \
                    WHEN itemid = 50806 THEN 'CHLORIDE' \
                    WHEN itemid = 50902 THEN 'CHLORIDE' \
                    WHEN itemid = 50809 THEN 'GLUCOSE' \
                    WHEN itemid = 50931 THEN 'GLUCOSE' \
                    WHEN itemid = 50810 THEN 'HEMATOCRIT' \
                    WHEN itemid = 51221 THEN 'HEMATOCRIT' \
                    WHEN itemid = 50811 THEN 'HEMOGLOBIN' \
                    WHEN itemid = 51222 THEN 'HEMOGLOBIN' \
                    WHEN itemid = 50813 THEN 'LACTATE' \
                    WHEN itemid = 51265 THEN 'PLATELET' \
                    WHEN itemid = 50822 THEN 'POTASSIUM' \
                    WHEN itemid = 50971 THEN 'POTASSIUM' \
                    WHEN itemid = 51275 THEN 'PTT' \
                    WHEN itemid = 51237 THEN 'INR' \
                    WHEN itemid = 51274 THEN 'PT' \
                    WHEN itemid = 50824 THEN 'SODIUM' \
                    WHEN itemid = 50983 THEN 'SODIUM' \
                    WHEN itemid = 51006 THEN 'BUN' \
                    WHEN itemid = 51300 THEN 'WBC' \
                    WHEN itemid = 51301 THEN 'WBC' \
                  ELSE null \
                END AS label \
              , CASE \
                  WHEN itemid = 50862 and valuenum >    10 THEN null \
                  WHEN itemid = 50868 and valuenum > 10000 THEN null \
                  WHEN itemid = 51144 and valuenum <     0 THEN null \
                  WHEN itemid = 51144 and valuenum >   100 THEN null \
                  WHEN itemid = 50882 and valuenum > 10000 THEN null \
                  WHEN itemid = 50885 and valuenum >   150 THEN null \
                  WHEN itemid = 50806 and valuenum > 10000 THEN null \
                  WHEN itemid = 50902 and valuenum > 10000 THEN null \
                  WHEN itemid = 50912 and valuenum >   150 THEN null \
                  WHEN itemid = 50809 and valuenum > 10000 THEN null \
                  WHEN itemid = 50931 and valuenum > 10000 THEN null \
                  WHEN itemid = 50810 and valuenum >   100 THEN null \
                  WHEN itemid = 51221 and valuenum >   100 THEN null \
                  WHEN itemid = 50811 and valuenum >    50 THEN null \
                  WHEN itemid = 51222 and valuenum >    50 THEN null \
                  WHEN itemid = 50813 and valuenum >    50 THEN null \
                  WHEN itemid = 51265 and valuenum > 10000 THEN null \
                  WHEN itemid = 50822 and valuenum >    30 THEN null \
                  WHEN itemid = 50971 and valuenum >    30 THEN null \
                  WHEN itemid = 51275 and valuenum >   150 THEN null \
                  WHEN itemid = 51237 and valuenum >    50 THEN null \
                  WHEN itemid = 51274 and valuenum >   150 THEN null \
                  WHEN itemid = 50824 and valuenum >   200 THEN null \
                  WHEN itemid = 50983 and valuenum >   200 THEN null \
                  WHEN itemid = 51006 and valuenum >   300 THEN null \
                  WHEN itemid = 51300 and valuenum >  1000 THEN null \
                  WHEN itemid = 51301 and valuenum >  1000 THEN null \
                ELSE le.valuenum \
                END AS valuenum \
              FROM icustays ie \
              LEFT JOIN labevents le \
                ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id \
                AND le.CHARTTIME between (ie.intime - interval '6' hour) and (ie.intime + interval '7' day)\
                AND le.ITEMID in \
                ( \
                  50868, \
                  50862, \
                  51144, \
                  50882, \
                  50885, \
                  50912, \
                  50902, \
                  50806, \
                  50931, \
                  50809, \
                  51221, \
                  50810, \
                  51222, \
                  50811, \
                  50813, \
                  51265, \
                  50971, \
                  50822, \
                  51275, \
                  51237, \
                  51274, \
                  50983, \
                  50824, \
                  51006, \
                  51301, \
                  51300  \
                ) \
                AND valuenum IS NOT null AND valuenum > 0 \
            ) pvt \
            GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id \
            ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id;"
     
    cursor.execute(view) 

def get_vitals_chart(cursor):
   
   # -- This query pivots the vital signs during the first 7 days of a patient's stay
   #-- Vital signs include heart rate, blood pressure, respiration rate, and temperature

    view = "DROP MATERIALIZED VIEW IF EXISTS vitalsfirstday CASCADE; \
            create materialized view vitalsfirstday as \
            SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id \
            , min(case when VitalID = 1 then valuenum else null end) as HeartRate_Min \
            , max(case when VitalID = 1 then valuenum else null end) as HeartRate_Max \
            , avg(case when VitalID = 1 then valuenum else null end) as HeartRate_Mean \
            , min(case when VitalID = 2 then valuenum else null end) as SysBP_Min \
            , max(case when VitalID = 2 then valuenum else null end) as SysBP_Max \
            , avg(case when VitalID = 2 then valuenum else null end) as SysBP_Mean \
            , min(case when VitalID = 3 then valuenum else null end) as DiasBP_Min \
            , max(case when VitalID = 3 then valuenum else null end) as DiasBP_Max \
            , avg(case when VitalID = 3 then valuenum else null end) as DiasBP_Mean \
            , min(case when VitalID = 4 then valuenum else null end) as MeanBP_Min \
            , max(case when VitalID = 4 then valuenum else null end) as MeanBP_Max \
            , avg(case when VitalID = 4 then valuenum else null end) as MeanBP_Mean \
            , min(case when VitalID = 5 then valuenum else null end) as RespRate_Min \
            , max(case when VitalID = 5 then valuenum else null end) as RespRate_Max \
            , avg(case when VitalID = 5 then valuenum else null end) as RespRate_Mean \
            , min(case when VitalID = 6 then valuenum else null end) as TempC_Min \
            , max(case when VitalID = 6 then valuenum else null end) as TempC_Max \
            , avg(case when VitalID = 6 then valuenum else null end) as TempC_Mean \
            , min(case when VitalID = 7 then valuenum else null end) as SpO2_Min \
            , max(case when VitalID = 7 then valuenum else null end) as SpO2_Max \
            , avg(case when VitalID = 7 then valuenum else null end) as SpO2_Mean \
            , min(case when VitalID = 8 then valuenum else null end) as Glucose_Min \
            , max(case when VitalID = 8 then valuenum else null end) as Glucose_Max \
            , avg(case when VitalID = 8 then valuenum else null end) as Glucose_Mean \
            FROM  ( \
            select ie.subject_id, ie.hadm_id, ie.icustay_id \
            , case \
              when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then 1 \
              when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then 2 \
              when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then 3 \
              when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 4 \
              when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then 5 \
              when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 6 \
              when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 6 \
              when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 7 \
              when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 8 \
              else null end as VitalID \
            , case when itemid in (223761,678) then (valuenum-32)/1.8 else valuenum end as valuenum \
            from icustays ie \
            left join chartevents ce \
            on ie.subject_id = ce.subject_id and ie.hadm_id = ce.hadm_id and ie.icustay_id = ce.icustay_id \
            and ce.charttime between ie.intime - interval '6' hour and ie.intime + interval '7' day \
            and ce.error IS DISTINCT FROM 1 \
            where ce.itemid in \
            ( \
            211, \
            220045, \
            51, \
            442, \
            455, \
            6701, \
            220179, \
            220050, \
            8368, \
            8440,  \
            8441,  \
            8555,  \
            220180,  \
            220051,  \
            456,  \
            52,  \
            6702,  \
            443,  \
            220052, \
            220181,  \
            225312,  \
            618, \
            615, \
            220210, \
            224690,  \
            646, 220277, \
            807, \
            811, \
            1529, \
            3745, \
            3744, \
            225664, \
            220621, \
            226537, \
            223762, \
            676, \
            223761, \
            678 \
            ) \
            ) pvt \
            group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id  \
            order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id;"
    
    cursor.execute(view)

def get_comorbidities(cursor):
    
    view="DROP MATERIALIZED VIEW IF EXISTS COMORBIDITIES CASCADE; \
        CREATE MATERIALIZED VIEW COMORBIDITIES AS \
        with icd as \
        ( \
          select hadm_id, seq_num, icd9_code \
          from diagnoses_icd \
          where seq_num != 1 \
        ) \
        , eliflg as \
        (\
        select hadm_id, seq_num, icd9_code\
        , CASE\
          when icd9_code in ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1\
          when SUBSTRING(icd9_code FROM 1 for 4) in ('4254','4255','4257','4258','4259') then 1\
          when SUBSTRING(icd9_code FROM 1 for 3) in ('428') then 1\
          else 0 end as CHF\
        , CASE \
          when icd9_code in ('42613','42610','42612','99601','99604') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('4260','4267','4269','4270','4271','4272','4273','4274','4276','4278','4279','7850','V450','V533') then 1 \
          else 0 end as ARRHY \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('0932','7463','7464','7465','7466','V422','V433') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('394','395','396','397','424') then 1 \
          else 0 end as VALVE \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('440','441') then 1 \
          else 0 end as PERIVASC \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('401') then 1 \
          else 0 end as HTN \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('402','403','404','405') then 1 \
          else 0 end as HTNCX \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2500','2501','2502','2503') then 1 \
          else 0 end as DM \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2504','2505','2506','2507','2508','2509') then 1 \
          else 0 end as DMCX \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2409','2461','2468') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('243','244') then 1 \
          else 0 end as HYPOTHY \
          , CASE \
          when icd9_code in ('40301','40311','40391','40402','40403','40412','40413','40492','40493') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('5880','V420','V451') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('585','586','V56') then 1 \
          else 0 end as RENLFAIL \
        , CASE \
          when icd9_code in ('07022','07023','07032','07033','07044','07054') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('0706','0709','4560','4561','4562','5722','5723','5724','5728','5733','5734','5738','5739','V427') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('570','571') then 1 \
          else 0 end as LIVER \
        , CASE \
          when icd9_code in ('72889','72930') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('446','714','720','725') then 1 \
          else 0 end as ARTH \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2871','2873','2874','2875') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('286') then 1 \
          else 0 end as COAG \
         , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2780') then 1 \
          else 0 end as OBESE \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2536') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('276') then 1 \
          else 0 end as LYTES \
        , CASE \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('2652','2911','2912','2913','2915','2918','2919','3030','3039','3050','3575','4255','5353','5710','5711','5712','5713','V113') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('980') then 1 \
          else 0 end as ALCOHOL \
        , CASE \
          when icd9_code in ('V6542') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 4) in ('3052','3053','3054','3055','3056','3057','3058','3059') then 1 \
          when SUBSTRING(icd9_code FROM 1 for 3) in ('292','304') then 1 \
          else 0 end as DRUG \
        from icd \
        )\
        , eligrp as \
        ( \
          select hadm_id \
          , max(chf) as chf \
          , max(arrhy) as arrhy \
          , max(valve) as valve \
          , max(perivasc) as perivasc \
          , max(htn) as htn \
          , max(htncx) as htncx \
          , max(renlfail) as renlfail \
          , max(dm) as dm \
          , max(dmcx) as dmcx \
          , max(hypothy) as hypothy \
          , max(liver) as liver \
          , max(arth) as arth \
          , max(coag) as coag \
          , max(obese) as obese \
          , max(lytes) as lytes \
          , max(alcohol) as alcohol \
          , max(drug) as drug \
          from eliflg \
        group by hadm_id \
        ) \
        select adm.hadm_id \
        , chf as CONGESTIVE_HEART_FAILURE \
        , arrhy as CARDIAC_ARRHYTHMIAS \
        , valve as VALVULAR_DISEASE \
        , perivasc as PERIPHERAL_VASCULAR \
        , renlfail as RENAL_FAILURE \
        , case \
            when htn = 1 then 1 \
            when htncx = 1 then 1 \
          else 0 end as HYPERTENSION \
        , case \
            when dmcx = 1 then 0 \
            when dm = 1 then 1 \
          else 0 end as DIABETES_UNCOMPLICATED \
        , dmcx as DIABETES_COMPLICATED \
        , hypothy as HYPOTHYROIDISM \
        , liver as LIVER_DISEASE \
        , obese as OBESITY \
        , alcohol as ALCOHOL_ABUSE \
        , drug as DRUG_ABUSE \
        from admissions adm \
        left join eligrp eli \
          on adm.hadm_id = eli.hadm_id \
        order by adm.hadm_id;"
        
    cursor.execute(view)

def count_icustays(cursor):
    
    query = "select * from icustays"
    cursor.execute(query) 
    
    rows = cursor.fetchall()
                 
if __name__ == '__main__':
    
    mimicDB="mimic"
    
    try:
        conn = psycopg2.connect(host="localhost",
                                user="postgres",
                                password="postgres",
                                database=mimicDB)
        cursor = conn.cursor()
    
    except Exception as error:
        print(error)
    
    test_postgres(cursor)
    
    urine_output(cursor)
    print("view urine_output created")
        
    weight_duration(cursor)
    print("view weight_duration created")
    
    urine_kidigo(cursor)
    print("view urine_kidigo created")
    
    creatinine(cursor)
    print("view creatinine created")
    
    kidigo_stages(cursor)
    print("view kidigo_stages created")
    query = "select * from kdigo_stages"
    df = pd.read_sql_query(query, conn)
    df.to_csv("AKI_KIDIGO_STAGES_SQL.csv", encoding='utf-8', header=True)
          
    kidigo_7_days(cursor)
    print("view kidigo_7_days created")     
    query = "select * from kdigo_stages_7day"
    df = pd.read_sql_query(query, conn)
    df.to_csv("AKI_KIDIGO_7D_SQL.csv", encoding='utf-8', header=True)
 
    kidigo_stages_creatinine(cursor)
    print("view kidigo_stages_creatinine created")
    query = "select * from kdigo_stages_creatinine"
    df = pd.read_sql_query(query, conn)
    df.to_csv("AKI_KIDIGO_STAGES_SQL_CREATININE.csv", encoding='utf-8', header=True)

    kidigo_7_days_creatinine(cursor)
    print("view kidigo_7_days_creatinine created")       
    query = "select * from kdigo_7_days_creatinine"
    df = pd.read_sql_query(query, conn)
    df.to_csv("AKI_KIDIGO_7D_SQL_CREATININE.csv", encoding='utf-8', header=True)

    get_labevents(cursor)
#      
    query = "select * from labstay"
    df = pd.read_sql_query(query, conn)   
    df.to_csv("labstay.csv", encoding='utf-8', header=True)
 
    get_vitals_chart(cursor)
     
    query = "select * from vitalsfirstday"
    df = pd.read_sql_query(query, conn)   
    df.to_csv("chart_vitals_stay.csv", encoding='utf-8', header=True)
     
    get_comorbidities(cursor)
   
    query = "select * from COMORBIDITIES"
    df = pd.read_sql_query(query, conn)   
    df.to_csv("comorbidities.csv", encoding='utf-8', header=True)

    count_icustays(cursor)
