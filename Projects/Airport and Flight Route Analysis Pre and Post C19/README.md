## Overview
This project focuses on analyzing and visualizing the trend of airport and airspace and in some cases, flight
routes before and after the Covid-19 pandemic struck. At the moment, the project is limited to regional
airspace (Chicago, New York, and Los Angeles) that have historically contained the densest and busiest air traffic over them.
The data used is historical data from the year 2019 to 2020, however; specific dates were chosen to generate heatmaps that visualize the density of the airspace over a specific area, defined by bounding box coordinates.
The dates used are '25th February 2020' for Pre-Covid, and '7th April 2020' for Post-Covid. All aircraft logged over the specified coordinates are filtered to be aircraft flying above or equal to 10,000 ft AGL, regardless of aircraft type, operator or type of flight. Commercial scheduled, commercial non-scheduled, cargo, and general aviation flights are also recorded.

## Tools and data:
Data on airports were obtained from the [Traffic-Vis tool] (https://traffic-viz.github.io/installation.html#updating-traffic), an API - aggregated from air traffic data from [Zenodo](https://zenodo.org/record/4893103). Flight routes and ADS-B information was accessed from [OpenSky](https://opensky-network.org/data/impala), which is used to generate the heatmaps; additional data may be procured using the Impala and Hadoop shells specified in the OpenSky usage guide.
Visualizations were generated in Python interactively using Altair, MatplotLib and Seaborn. The geo-spatial visuals were generated in Tableau, uploaded to the Public server and embedded into the notebook within this folder.
If you need the data - use the link embeds in this markdown to navigate to the sources. OpenSky needs an access key for you to query data relating to a specific flight or flight route, so you need to fill in a form to request access to it.

- Warning: The data on Zenodo from 2019 to 2020 is approximately 4 GB in size - do not download if you do not have the space on your hard drive.

## IMPORTANT: Viewing the notebook:
Since GitHub scrubs all Javascript embeddings, you won't be able to interact with or view any of the plots - instead, I recommend you use [nbviewer](https://nbviewer.jupyter.org/) to look at them. Simply paste the URL of the notebook in GitHub and nbviewer will display all outputs as intended.

Thanks for visiting!
