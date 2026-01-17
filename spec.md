# Overview

Revision 1

We're likely going to be changing input a lot. Given the proposed data access pattern from the [Model V2 Diagram](<Model V2.drawio.pdf>)

# Spec 

We want this to be as API friendly as possible. The primary motivating factor is that we take in a config file that contains an array of json strings to tell the *data retrieval service* to pull back. 

## Request Body 1: Data retrieval request
```json
{ 
    batter: string
    pitcher: string
    features: [string...]
}
```
For the sake of developer convenience, available features should be accessible through an API exposed from the Data Retrieval service that returns a list of `features`. The data retrieval service will guarantee that all the elements in the `features` JSON array are valid. We can set up a configuration file that matches our database columns.

### Response
We have mixed features. Ideally, we should return each feature bucketed into the kinds of features they would be embedded as since there are different techniques: 

1. Categorical columns. This includes things like `pitch_name`; things that have definite state. We can also identify the batters and pitchers by ID here. 
2.  Boolean columns. This includes binary states, such as people being on base, the result of the last at bat, or other things that we want to embed with a yes/no response.

```json
{
    categorical: [
        { 
            feature_name: string
            value: string?
        },
        ...
    ],
    boolean: [
        {
            feature_name: string 
            value: boolean
        }, 
        ...
    ] 
}
```

## Request Body 2: Request to model
We want the request to the model to be as customizable as possible. We can likely chop this step and push straight from the data retrieval service to the model using the specification described above.

### Response (from the model)
The response from the model is something that we'll want to render for the frontend. It would be most convenient if each iteration from the model was pushed into nodes that compose the entire sequence. Each node would just have every pitch type as keys and their recommendation weights as values

```json
{ 
    pitch_one: { 
        CH: float, // this would be a float type representing "probability" or weight of recommendation to the user 
        CU: float, 
        FC: float, 
        EP: float, 
        FO: float, 
        FF: float, 
        KN: float, 
        KC: float, 
        SC: float, 
        SI: float, 
        SL: float, 
        SV: float, 
        FS: float, 
        ST: float // all the pitch nodes will look like this
    }, 
    pitch_two: { 
        (see above)
    }, 
    pitch_three: {
        (see above) 
    }, 
    pitch_four: { 
        (see above)
    }
}
```