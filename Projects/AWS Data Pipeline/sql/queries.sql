CREATE TABLE IF NOT EXISTS test_table(
  iso_code varchar(15),
  continent varchar(50),
  location varchar(50),
  date date,
  total_cases int,
  new_cases int,
  total_cases_per_million float,
  new_cases_per_million float,
  icu_patients int,
  icu_patients_per_million float,
  hosp_patients int,
  hosp_patients_per_million float,
  new_tests int,
  new_tests_per_thousand float,
  tests_units varchar(50),
  total_boosters int,
  stringency_index float,
  population int,
  population_density float,
  median_age float,
  aged_65_older float,
  gdp_per_capita float,
  extreme_poverty float,
  cardiovasc_death_rate float,
  diabetes_prevalence float,
  female_smokers float,
  male_smokers float,
  handwashing_facilities float,
  life_expectancy float,
  human_development_index float,
  excess_mortality float,
  total_deaths int,
  new_deaths int,
  total_deaths_per_million float,
  new_deaths_per_million float,
  reproduction_rate float,
  total_vaccinations int,
  people_vaccinated int,
  people_fully_vaccinated int,
  new_vaccinations int,
  total_tests int,
  positive_rate float,
  tests_per_case float
  )

  -- View table
  SELECT * FROM test_table
  LIMIT 100;

  -- Remove header row because I'm an idiot
  DELETE FROM test_table
  WHERE iso_code = 'iso_code';

  -- GLOBAL analysis
  -- death rates
  SELECT location, MAX(total_deaths) AS death_count
  FROM test_table
  WHERE continent IS NOT NULL AND location NOT IN ('World', 'Upper middle income', 'High income', 'Lower middle income', 'Europe', 'Asia', 'North America', 'South America', 'European Union', 'International', 'Low income')
  GROUP BY location
  ORDER BY death_count DESC;

  -- Death toll by continent
  SELECT continent, MAX(total_deaths) AS death_toll
  FROM test_table
  WHERE continent IS NOT NULL
  GROUP BY continent
  ORDER BY death_toll DESC;

-- United States
-- Create table for United States
CREATE TABLE united_states AS
SELECT * FROM test_table
WHERE location = 'United States';

-- Cases trend
SELECT location, date, new_cases
FROM united_states
WHERE date BETWEEN '2020-1-22' AND '2022-1-22'
ORDER BY 1,2;

-- examining vaccinations and cases
SELECT date, total_cases, total_vaccinations, total_deaths
FROM united_states
ORDER BY total_vaccinations;

-- total deaths in the US
SELECT MAX(total_deaths) as death_toll
FROM united_states;

-- case count and percentage of population in the US with stupid RS format
SELECT population, MAX(total_cases) as case_count, 100.0 * case_count / population as percentage
FROM united_states
GROUP BY population
ORDER BY case_count;

-- looking at tests and cases
SELECT date, new_cases, new_tests, new_deaths
FROM united_states;

-- same query but replacing nulls
SELECT date, NVL(new_cases, 0.0) as cases, NVL(new_tests, 0.0) as tests, NVL(new_deaths, 0.0) as deaths
FROM united_states
ORDER BY date ASC;
