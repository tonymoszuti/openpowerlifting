CREATE TABLE IF NOT EXISTS powerlifting_results_raw (
    Name VARCHAR(255),
    Sex CHAR(2),
    Event VARCHAR(255),
    Equipment VARCHAR(255),
    Age VARCHAR(255),
    AgeClass VARCHAR(255),
    BirthYearClass VARCHAR(255),
    Division VARCHAR(255),
    BodyweightKg DECIMAL(10, 2),
    WeightClassKg VARCHAR(255),
    Squat1Kg DECIMAL(10, 2),
    Squat2Kg DECIMAL(10, 2),
    Squat3Kg DECIMAL(10, 2),
    Squat4Kg DECIMAL(10, 2),
    Best3SquatKg DECIMAL(10, 2),
    Bench1Kg DECIMAL(10, 2),
    Bench2Kg DECIMAL(10, 2),
    Bench3Kg DECIMAL(10, 2),
    Bench4Kg DECIMAL(10, 2),
    Best3BenchKg DECIMAL(10, 2),
    Deadlift1Kg DECIMAL(10, 2),
    Deadlift2Kg DECIMAL(10, 2),
    Deadlift3Kg DECIMAL(10, 2),
    Deadlift4Kg DECIMAL(10, 2),
    Best3DeadliftKg DECIMAL(10, 2),
    TotalKg DECIMAL(10, 2),
    Place VARCHAR(255),
    Dots DECIMAL(10, 2),
    Wilks DECIMAL(10, 2),
    Glossbrenner DECIMAL(10, 2),
    Goodlift VARCHAR(255),
    Tested BOOLEAN,
    Country VARCHAR(255),
    State VARCHAR(255),
    Federation VARCHAR(255),
    ParentFederation VARCHAR(255),
    Date DATE,
    MeetCountry VARCHAR(255),
    MeetState VARCHAR(255),
    MeetTown VARCHAR(255),
    MeetName VARCHAR(255),
    Sanctioned BOOLEAN
);

COPY powerlifting_results_raw
FROM PROGRAM 'cat /tmp/data/*.csv'
WITH (FORMAT csv, HEADER);

CREATE TABLE powerlifting_results_i AS
SELECT *, 'i' AS mode
FROM powerlifting_results_raw;
	
	
UPDATE powerlifting_results_i
SET mode = CASE
            WHEN Sex NOT IN ('M', 'F', 'Mx') THEN 'D' -- if sex is not M,F,Mx set mode to 'D'
            when equipment not in ('Raw') then 'D'
        	WHEN place = 'NS' THEN 'D' -- if place is marked as 'NS' set mode to 'D'
      		WHEN tested != 'yes' THEN 'D'
            WHEN sanctioned != 'yes' THEN 'D'
            WHEN TRIM(weightclasskg) NOT IN ('47', '52', '57', '63', '69', '76', '84', '84+',
                                             '59', '66', '74', '83', '93', '105', '120', '120+') THEN 'D' -- if weight class is not one of the specified values set mode to 'D'
            
            ELSE mode
          END
WHERE mode = 'i'; -- Add the WHERE clause here to target only rows where mode is currently 'i'

CREATE TABLE powerlifting_results_final AS
SELECT
    Name,
    Sex,
    Event,
    Equipment,
    CASE 
        WHEN Age::text !~ '^\d+(\.\d+)?$' THEN NULL -- If Age is not a valid number, set to NULL
        ELSE ROUND(Age::numeric) -- Otherwise, round the age
    END AS Age,
    AgeClass,
    BirthYearClass,
    Division,
    BodyweightKg,
    WeightClassKg,
    Squat1Kg,
    Squat2Kg,
    Squat3Kg,
    Squat4Kg,
    Best3SquatKg,
    Bench1Kg,
    Bench2Kg,
    Bench3Kg,
    Bench4Kg,
    Best3BenchKg,
    Deadlift1Kg,
    Deadlift2Kg,
    Deadlift3Kg,
    Deadlift4Kg,
    Best3DeadliftKg,
    TotalKg,
    Place,
    Dots, 
    Wilks,
    Goodlift,
    Tested,
    Country,
    State,
    Federation,
    ParentFederation,
    Date,
    MeetCountry,
    MeetState,
    MeetTown,
    MeetName,
    Sanctioned
FROM powerlifting_results_i
WHERE mode = 'i';