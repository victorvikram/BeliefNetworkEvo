# Guide to Cleaning GSS Data

This guide explains how the GSS (General Social Survey) data cleaning pipeline works. The process transforms raw GSS data into two standardized versions suitable for analysis.

## Overview

The cleaning pipeline consists of three main steps:
1. Data import from raw SAS files
2. Data filtering and normalization 
3. Creation of two cleaned versions:
   - Regular version: Variables normalized to [-1, 1] with natural zero points preserved
   - Median-centered version: Variables normalized to [-1, 1] and centered around their median (this will be used for calculating frustration)

## Data Import

The `import_gss.py` script handles raw data import:
- You should run this script to import the raw data into a pandas dataframe. It will also cache the data so that it can be loaded faster next time.
- You can then access the data any time by calling `df, metadata = import_dataset()`

## Data Cleaning Process

We need to clean the data as well. 

The raw data has lots of variables we don't need. It's also not normalised between -1 and 1, which is a bit annoying, though not essential.
And there are many "derived variables" that we will want to create. These are things like "Voted republican" or "Voted in last election".

So, by "cleaning" the data, we will do all these things.

### Variable Types

The basic problem that we need to solve is that survey questions all have different scales.

There are two categories of questions:

Symmetric questions and asymmetric questions.

- Symmetric questions have a natural zero point in the middle of the scale.
- Asymmetric questions have a natural zero point at one end of the scale.

An example of a symmetric question is "To what extent do you consider yourself a liberal versus a conservative?" and the responses space a 1-7 scale, with 4 being some sort of centre stance.

An example of an asymmetric question is "Do you identify as a Catholic?" and the responses space a 1-2 scale, with 1 being "no" and 2 being "yes".

Another example of an asymmetric question is "Who did you vote for in the 68' election?" and the responses are several presidents and "didn't vote". This cannot be mapped to any single scale and instead 
needs to be mapped to several binary yes/no variables (i.e., "voted for Nixon", "voted for Humphrey", "voted for Wallace", "didn't vote"), each of which must be mapped to [0, 1].

A final example of an asymmetric question is "How many hours per day do you watch TV?" and the responses are 1-24. This of course can only be mapped to [0, 1] because there is no natural zero point (that is, until we define the zero point as the median value).

In the former case, the zero point is natural and in the middle of the scale. We are therefore able to normalise the data to [-1, 1].

In the latter case, mapping to [-1, 1] is not natural because zero wouldn't correspond to anything (there is no "not sure" option). We therefore map this to [0, 1] instead.

This mapping strategy is up for debate. If you have ideas to improve this, or maybe you disagree with our approach, let's all discuss it.

> A concern I have here is that some questions, like...
> "Do you think that Blacks have worse job opportunities than Whites due to discrimination?" with yes/no responses.
> We could map this to [-1, 1] but it would be a bit weird because there is no "neutral" option that would correspond to zero -- there is an "I don't know" option that would correspond to zero.
HOWEVER, many questions that do have a natural zero ALSO have an "I don't know" option. So, this would be a bit inconsistent of us. 

>We could always argue that IF there is a natural zero, we use it. And then the next best thing is the "I don't know" option. How does this sound?

So, in summary, we will:

- Normalise symmetric questions to [-1, 1]
- Normalise asymmetric questions to [0, 1]


### Cleaning Steps

1. **Initial Filtering**
   - Removes unwanted columns
   - Handles missing values
   - Validates data types

2. **Regular Version Transformation**
   - Preserves natural zero points
   - Maps categorical responses to normalized values
   - Maintains interpretable scales

3. **Median-Centered Version**
   - Centers all variables around their median
   - Normalizes to [-1, 1] range
   - Better for certain statistical analyses

## Variable Categories

The cleaning process handles these main variable categories:

### Opinion Variables
- Political views
- Trust measures
- Fairness perceptions
- Party identification

### Frequency Measures
- Religious attendance
- TV watching hours
- News following frequency

### Confidence Measures
- Confidence in institutions
- Trust in organizations
- Faith in systems

### Support Measures
- Government spending preferences
- Program support
- Policy positions

## Data Validation

The `validate_cleaned_datasets.py` script performs comprehensive checks:

1. **Range Validation**
   - Ensures all values are within [-1, 1]
   - Checks for outliers
   - Validates transformations

2. **Consistency Checks**
   - Compares both cleaned versions
   - Verifies transformations
   - Checks data completeness

3. **Quality Assessment**
   - Generates visualization plots
   - Produces completeness reports
   - Validates year spans

## Usage Notes

1. **Caching**
   - Both raw and cleaned data are cached
   - Check cached_data directory for existing files
   - Use `force_reload=True` to bypass cache

2. **Error Handling**
   - Missing values preserved as NaN
   - Warnings issued for unmapped values
   - Data type mismatches flagged

3. **Performance**
   - Uses vectorized operations
   - Efficient memory management
   - Cached results for speed

## Best Practices

1. Always validate cleaned data before use
2. Check for warnings about unmapped values
3. Understand which version (regular or median-centered) is appropriate
4. Document any additional cleaning steps
5. Verify transformations maintain meaningful relationships

## Common Issues

1. **Missing Values**
   - Some variables have high missing rates
   - Consider imputation strategies
   - Document missing patterns

2. **Scale Inversions**
   - Check variable directions match expectations
   - Verify coding of negative/positive responses
   - Document any scale reversals

3. **Temporal Changes**
   - Survey questions change over time
   - Response options may vary
   - Consider year-specific adjustments

## References

- GSS Codebook
- Variable Documentation
- Cleaning Configuration (DataConfig class)
- Validation Reports
