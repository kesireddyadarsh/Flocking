//
//  main.cpp
//  HOF
//
//  Created by adarsh kesireddy on 6/19/17.
//  Copyright © 2017 AK. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <complex>


using namespace std;

bool run_simulation = true;
bool test_simulation = true;

/********************************************************
 Variables for flocking. Make modification after implementation
 ********************************************************/

#define PI 3.14159265
#define d 7
#define r 1.2*d
#define r_prime 0.6* r
#define kappa 1.2
#define d_p 0.6*d
#define r_p 1.2*d_p
#define epsilon 0.1
#define aa 5
#define bb aa
#define h_a 0.2
#define h_b 0.9

const double c1_b = 10;
const double c2_b = 2*sqrt(c1_b);
const double c1_g = 0.2*c1_b;
const double c2_g = 2*sqrt(c1_g);
const double c1_a = 0.5*c1_g;
const double c2_a = 2*sqrt(c1_a);

const vector<double> qd={200,30};
const vector<double> pd={5,0};


/*************************
 Neural Network
 ************************/

struct connect{
    double weight;
};

static double random_global(double a) { return a* (rand() / double(RAND_MAX)); }

// This is for each Neuron
class Neuron;
typedef vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    vector<connect> z_outputWeights;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    unsigned z_myIndex;
    double z_outputVal;
    void setOutputVal(double val) { z_outputVal = val; }
    double getOutputVal(void) const { return z_outputVal; }
    void feedForward(const Layer prevLayer);
    double transferFunction(double x);
    
};

//This creates connection with neurons.
Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight() - 0.5;
    }
    z_myIndex = myIndex;
}

double Neuron::transferFunction(double x){
    
    int case_to_use = 1;
    switch (case_to_use) {
        case 1:
            return tanh(x);
            break;
        case 2:
            //Dont use this case
            return 1/(1+exp(x));
            break;
        case 3:
            return x/(1+abs(x));
            break;
            
        default:
            break;
    }
    
    return tanh(x);
}

void Neuron::feedForward(const Layer prevLayer){
    double sum = 0.0;
    bool debug_sum_flag = false;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        if(debug_sum_flag == true){
            cout<<prevLayer[n].getOutputVal()<<endl;
            cout<<&prevLayer[n].z_outputWeights[z_myIndex];
            cout<<prevLayer[n].z_outputWeights[z_myIndex].weight;
        }
        sum += prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight;
        //cout<<"This is sum value"<<sum<<endl;
    }
    z_outputVal = Neuron::transferFunction(sum);
}

//This is single neural network
class Net{
public:
    Net(vector<unsigned> topology);
    void feedForward(vector<double> inputVals);
    vector<Layer> z_layer;
    vector<double> outputvaluesNN;
    double backProp();
    double z_error;
    double z_error_temp;
    vector<double> z_error_vector;
    void mutate();
    vector<double> temp_inputs;
    vector<double> temp_targets;
    
    //flocking
    double fitness;
    double best_distance;
    vector<double> distance_values;
    
};

Net::Net(vector<unsigned> topology){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>15){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        
        z_layer.push_back(Layer());
        
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
    }
}

void Net::mutate(){
    /*
     //popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer.at(l).size(); n++) {
            for (int z=0 ; z< z_layer.at(l).at(n).z_outputWeights.size(); z++) {
                z_layer.at(l).at(n).z_outputWeights.at(z).weight += random_global(.5)-random_global(.5);
            }
        }
    }
}

void Net::feedForward(vector<double> inputVals){
    
    assert(inputVals.size() == z_layer[0].size()-1);
    for (unsigned i=0; i<inputVals.size(); ++i) {
        z_layer[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
        Layer &prevLayer = z_layer[layerNum - 1];
        for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
            z_layer[layerNum][n].feedForward(prevLayer);
        }
    }
    temp_inputs.clear();
    
    
    Layer &outputLayer = z_layer.back();
    z_error_temp = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        //cout<<"This is value from outputlayer.getourputvalue:::::"<<outputLayer[n].getOutputVal()<<endl;
        //double delta = temp_targets[n] - outputLayer[n].getOutputVal();
        //cout<<"This is delta value::"<<delta;
        //z_error_temp += delta * delta;
        outputvaluesNN.push_back(outputLayer[n].getOutputVal());
    }
    
}

double Net::backProp(){
    z_error = 0.0;
    for (int temp = 0; temp< z_error_vector.size(); temp++) {
        //cout<<"This is z_error_vector"<<temp<<" value::"<< z_error_vector[temp]<<endl;
        z_error += z_error_vector[temp];
    }
    //    cout<<"This is z_error::"<<z_error<<endl;
    return z_error;
}

/***********************
 POI
 **********************/
class POI{
public:
    double x_position_poi,y_position_poi,value_poi;
    //Environment test;
    //vector<Rover> individualRover;
    vector<double> x_position_poi_vec;
    vector<double> y_position_poi_vec;
    vector<double> value_poi_vec;
};

/************************
 Environment
 ***********************/

class Environment{
public:
    vector<POI> individualPOI;
    vector<POI> group_1;
    vector<POI> group_2;
};

/************************
 Rover
 ***********************/

double resolve(double angle);


class Rover{
    //Environment environment_object;
public:
    double x_position,y_position;
    double p_x,p_y;
    double previous_x_position, previous_y_position;
    vector<double> x_position_vec,y_position_vec;
    vector<double> sensors;
    vector<Net> singleneuralNetwork;
    void sense_poi(double x, double y, double val);
    void sense_rover(double x, double y);
    double sense_poi_delta(double x_position_poi,double y_position_poi);
    double sense_rover_delta(double x_position_otherrover, double y_position_otherrover);
    vector<double> controls;
    double delta_x,delta_y;
    double theta;
    double previous_theta;
    double phi;
    void reset_sensors();
    int find_quad(double x, double y);
    double find_phi(double x, double y);
    double find_theta(double x_sensed, double y_sensed);
    void move_rover(double dx, double dy);
    double reward =0.0;
    void sense_all_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover);
    
    //stored values
    vector<double> max_reward;
    vector<double> policy;
    //vector<double> best_closest_distance;
    
    //Neural network
    vector<Net> network_for_agent;
    void create_neural_network_population(int numNN,vector<unsigned> topology);
    vector<Net> path_finder_network;
    void create_path_network(int numNN,vector<unsigned>* p_topology);
    
    //random numbers for neural networks
    vector<int> random_numbers;
    
    //Check for leader
    bool leader = false;
    bool move_possible = false;
    
    //This to calculate velocity and time step
    vector<double> velocity;
    
    //Fitness
    double blockage;
    double collision;
    double flocking;
    
    //velocity
    double velocity_of_agent;
    double velocity_of_agent_x;
    double velocity_of_agent_y;
    
    //distance between rovers
    vector<double> vec_distance_between_agents;
    
    
};

// variables used: indiNet -- object to Net
void Rover::create_neural_network_population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology);
        network_for_agent.push_back(singleNetwork);
    }
    
}

void Rover::create_path_network(int numNN,vector<unsigned>* p_topology){
    for (int neural_network = 0 ; neural_network < numNN; neural_network++) {
        Net singleNetwork(*p_topology);
        path_finder_network.push_back(singleNetwork);
    }
}

//Function returns: sum of values of POIs divided by their distance
double Rover::sense_poi_delta(double x_position_poi,double y_position_poi ){
    double delta_sense_poi=0;
    double distance = sqrt(pow(x_position-x_position_poi, 2)+pow(y_position-y_position_poi, 2));
    double minimum_observation_distance =0.0;
    delta_sense_poi=(distance>minimum_observation_distance)?distance:minimum_observation_distance ;
    return delta_sense_poi;
}

//Function returns: sum of sqaure distance from a rover to all the other rovers in the quadrant
double Rover::sense_rover_delta(double x_position_otherrover, double y_position_otherrover){
    double delta_sense_rover=0.0;
    if (x_position_otherrover == NULL || y_position_otherrover == NULL) {
        return delta_sense_rover;
    }
    double distance = sqrt(pow(x_position-x_position_otherrover, 2)+pow(y_position-y_position_otherrover, 2));
    delta_sense_rover=(1/distance);
    
    return delta_sense_rover;
}

void Rover::sense_poi(double poix, double poiy, double val){
    double delta = sense_poi_delta(poix, poiy);
    int quad = find_quad(poix,poiy);
    sensors.at(quad) += val/delta;
}

void Rover::sense_rover(double otherx, double othery){
    double delta = sense_poi_delta(otherx,othery);
    int quad = find_quad(otherx,othery);
    sensors.at(quad+4) += 1/delta;
}

void Rover::reset_sensors(){
    sensors.clear();
    for(int i=0; i<8; i++){
        sensors.push_back(0.0);
    }
}

double Rover::find_phi(double x_sensed, double y_sensed){
    double distance_in_x_phi =  x_sensed - x_position;
    double distance_in_y_phi =  y_sensed - y_position;
    double deg2rad = 180/PI;
    double phi = (atan2(distance_in_x_phi,distance_in_y_phi) *(deg2rad));
    
    return phi;
}

double Rover::find_theta(double x_sensed, double y_sensed){
    double distance_in_x_theta =  x_sensed - x_position;
    double distance_in_y_theta =  y_sensed - y_position;
    theta += atan2(distance_in_x_theta,distance_in_y_theta) * (180 / PI);
    
    return phi;
}

int Rover::find_quad(double x_sensed, double y_sensed){
    int quadrant;
    
    double phi = find_phi(x_sensed, y_sensed);
    double quadrant_angle = phi - theta;
    quadrant_angle = resolve(quadrant_angle);
    assert(quadrant_angle != NAN);
    //    cout << "IN QUAD: FIND PHI: " << phi << endl;
    
    phi = resolve(phi);
    
    //    cout << "IN QUAD: FIND PHI2: " << phi << endl;
    
    int case_number;
    if ((0 <= quadrant_angle && 45 >= quadrant_angle)||(315 < quadrant_angle && 360 >= quadrant_angle)) {
        //do something in Q1
        case_number = 0;
    }else if ((45 < quadrant_angle && 135 >= quadrant_angle)) {
        // do something in Q2
        case_number = 1;
    }else if((135 < quadrant_angle && 225 >= quadrant_angle)){
        //do something in Q3
        case_number = 2;
    }else if((225 < quadrant_angle && 315 >= quadrant_angle)){
        //do something in Q4
        case_number = 3;
    }
    quadrant = case_number;
    
    //    cout << "QUADANGLE =  " << quadrant_angle << endl;
    //    cout << "QUADRANT = " << quadrant << endl;
    
    return quadrant;
}

void Rover::move_rover(double dx, double dy){
    previous_x_position = x_position;
    previous_y_position = y_position;
    previous_theta = theta;
    double aom = atan2(dy,dx)*180/PI; /// angle of movement
    double rad2deg = PI/180;
    x_position = x_position + sin(theta*rad2deg) * dy + cos(theta*rad2deg) * dx;
    y_position = y_position + sin(theta*rad2deg) * dx + cos(theta*rad2deg) * dy;
    theta = theta + aom;
    theta = resolve(theta);
    
    //x_position =(x_position)+  (dy* cos(theta*(PI/180)))-(dx *sin(theta*(PI/180)));
    //y_position =(y_position)+ (dy* sin(theta*(PI/180)))+(dx *cos(theta*(PI/180)));
    //theta = theta+ (atan2(dx,dy) * (180 / PI));
    //theta = resolve(theta);
}


//Takes all poi values and update sensor values
void Rover::sense_all_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover){
    reset_sensors();
    
    double temp_delta_value = 0.0;
    vector<double> temp_delta_vec;
    int temp_quad_value =0;
    vector<double> temp_quad_vec;
    
    assert(x_position_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    assert(value_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    
    for (int value_calculating_delta = 0 ; value_calculating_delta < x_position_poi_vec_rover.size(); value_calculating_delta++) {
        temp_delta_value = sense_poi_delta(x_position_poi_vec_rover.at(value_calculating_delta), y_position_poi_vec_rover.at(value_calculating_delta));
        temp_delta_vec.push_back(temp_delta_value);
    }
    
    for (int value_calculating_quad = 0 ; value_calculating_quad < x_position_poi_vec_rover.size(); value_calculating_quad++) {
        temp_quad_value = find_quad(x_position_poi_vec_rover.at(value_calculating_quad), y_position_poi_vec_rover.at(value_calculating_quad));
        temp_quad_vec.push_back(temp_quad_value);
    }
    
    assert(temp_delta_vec.size()== temp_quad_vec.size());
    
    for (int update_sensor = 0 ; update_sensor<temp_quad_vec.size(); update_sensor++) {
        sensors.at(temp_quad_vec.at(update_sensor)) += value_poi_vec_rover.at(update_sensor)/temp_delta_vec.at(update_sensor);
    }
    
}

/*************************
 Population
 ************************/
//This is for population of neural network
class Population{
public:
    void create_Population(int numNN,vector<unsigned> topology);
    vector<Net> popVector;
    void runNetwork(vector<double> inputVal,int number_neural);
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    int returnIndex(int numNN);
    void repop(int numNN);
    
};

// variables used: indiNet -- object to Net
void Population::create_Population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology);
        popVector.push_back(singleNetwork);
    }
    
}

//Return index of higher
int Population::returnIndex(int numNN){
    int temp = numNN;
    int number_1 = (rand() % temp);
    int number_2 = (rand() % temp);
    while (number_1 == number_2) {
        number_2 = (rand() % temp);
    }
    
    if (popVector[number_1].z_error<popVector[number_2].z_error) {
        return number_2;
    }else if (popVector[number_1].z_error>popVector[number_2].z_error){
        return number_1;
    }else{
        return NULL;
    }
}

void Population::repop(int numNN){
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% popVector.size();
        popVector.push_back(popVector.at(R));
        popVector.back().mutate();
    }
}

void Population::runNetwork(vector<double> inputVals,int num_neural){
    popVector.at(num_neural).feedForward(inputVals);
    popVector.at(num_neural).backProp();
}

/**************************
 Simulation Functions
 **************************/
// Will resolve angle between 0 to 360
double resolve(double angle){
    while(angle >= 360){
        angle -=360;
    }
    while(angle < 0){
        angle += 360;
    }
    while (angle == 360) {
        angle = 0;
    }
    return angle;
}


double find_scaling_number(vector<Rover>* teamRover, POI* individualPOI){
    double number =0.0;
    double temp_number =0.0;
    vector < vector <double> > group_sensors;
    
    for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number< individualPOI->value_poi_vec.size(); policy_number++) {
            teamRover->at(rover_number).reset_sensors();
            teamRover->at(rover_number).sense_poi(individualPOI->x_position_poi_vec.at(policy_number), individualPOI->y_position_poi_vec.at(policy_number), individualPOI->value_poi_vec.at(policy_number));
            group_sensors.push_back(teamRover->at(rover_number).sensors);
        }
    }
    
    assert(!group_sensors.empty());
    
    for (int i=0; i<group_sensors.size(); i++) {
        temp_number=*max_element(group_sensors.at(i).begin(), group_sensors.at(i).end());
        if (temp_number>number) {
            number=temp_number;
        }
    }
    
    assert(number != 0.0);
    return number;
}


void remove_lower_fitness_network(Population * p_Pop,vector<Rover>* p_rover){
    
    bool VERBOSE = false;
    
    //evolution
    double temp_selection_number= p_Pop->popVector.size()/2; //select half the size
    for (int selectNN=0; selectNN<(temp_selection_number); ++selectNN) {
        double temp_random_1 = rand()%p_Pop->popVector.size();
        double temp_random_2 = rand()%p_Pop->popVector.size();
        while(temp_random_1==temp_random_2) {
            temp_random_2 = rand()%p_Pop->popVector.size();
        }
        double random_rover_number = rand()%p_rover->size();
        
        if (p_rover->at(random_rover_number).max_reward.at(temp_random_1)>p_rover->at(random_rover_number).max_reward.at(temp_random_2)) {
            //delete neural network temp_random_2
            p_Pop->popVector.erase(p_Pop->popVector.begin()+temp_random_2);
            p_rover->at(random_rover_number).max_reward.erase(p_rover->at(random_rover_number).max_reward.begin()+temp_random_2);
        }else{
            //delete neural network temp_random_1
            p_Pop->popVector.erase(p_Pop->popVector.begin()+temp_random_1);
            p_rover->at(random_rover_number).max_reward.erase(p_rover->at(random_rover_number).max_reward.begin()+temp_random_1);
        }
        
        
    }
    
    //clear maximum values
    for (int clear_max_vec =0 ; clear_max_vec<p_rover->size(); clear_max_vec++) {
        p_rover->at(clear_max_vec).max_reward.clear();
    }
    
    if(VERBOSE){
        cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"<<endl;
        for (int temp_print =0 ; temp_print<p_rover->size(); temp_print++) {
            cout<<"This is size of max reward::"<<p_rover->at(temp_print).max_reward.size()<<endl;
        }
    }
}

void repopulate_neural_networks(int numNN,Population* p_Pop){
    vector<unsigned> a;
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% p_Pop->popVector.size();
        Net N(a);
        N=p_Pop->popVector.at(R);
        N.mutate();
        p_Pop->popVector.push_back(N);
    }
}

/*****************************************************************
 Test Rover in environment
 ***************************************************************/

// Tests Stationary POI and Stationary Rover in all directions
bool POI_sensor_test(){
    bool VERBOSE = false;
    
    bool passfail = false;
    
    bool pass1 = false;
    bool pass2 = false;
    bool pass3 = false;
    bool pass4 = false;
    
    POI P;
    Rover R;
    
    /// Stationary Rover
    R.x_position = 0;
    R.y_position = 0;
    R.theta = 0; /// north
    
    P.value_poi = 10;
    
    /// POI directly north, sensor 0 should read; no others.
    P.x_position_poi = 0.001;
    P.y_position_poi = 1;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) != 0 && R.sensors.at(1) == 0 && R.sensors.at(2) ==0 && R.sensors.at(3) == 0){
        pass1 = true;
    }
    
    assert(pass1 == true);
    
    if(VERBOSE){
        cout << "Direct north case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    /// POI directly south, sensor 2 should read; no others.
    P.x_position_poi = 0;
    P.y_position_poi = -1;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) == 0 && R.sensors.at(2) !=0 && R.sensors.at(3) == 0){
        pass2 = true;
    }
    
    assert(pass2 == true);
    
    if(VERBOSE){
        cout << "Direct south case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    /// POI directly east, sensor 1 should read; no others.
    P.x_position_poi = 1;
    P.y_position_poi = 0;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) != 0 && R.sensors.at(2) ==0 && R.sensors.at(3) == 0){
        pass3 = true;
    }
    
    assert(pass3 == true);
    
    if(VERBOSE){
        cout << "Direct east case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    
    /// POI directly west, sensor 3 should read; no others.
    P.x_position_poi = -1;
    P.y_position_poi = 0;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) == 0 && R.sensors.at(2) ==0 && R.sensors.at(3) != 0){
        pass4 = true;
    }
    
    if(VERBOSE){
        cout << "Direct west case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    assert(pass4 == true);
    
    
    if(pass1 && pass2 && pass3 && pass4){
        passfail = true;
    }
    assert(passfail == true);
    return passfail;
}

//Test for stationary rovers test in all directions
bool rover_sensor_test(){
    bool passfail = false;
    
    bool pass5 = false;
    bool pass6 = false;
    bool pass7 = false;
    bool pass8 = false;
    
    Rover R1;
    Rover R2;
    R1.x_position = 0;
    R1.y_position = 0;
    R1.theta = 0; // north
    R2.theta = 0;
    
    // case 1, Rover 2 to the north
    R2.x_position = 0;
    R2.y_position = 1;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 4 should fire, none other.
    if(R1.sensors.at(4) != 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) == 0){
        pass5 = true;
    }
    assert(pass5 == true);
    
    // case 2, Rover 2 to the east
    R2.x_position = 1;
    R2.y_position = 0;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 5 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) != 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) == 0){
        pass6 = true;
    }
    assert(pass6 == true);
    
    // case 3, Rover 2 to the south
    R2.x_position = 0;
    R2.y_position = -1;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 6 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) != 0 && R1.sensors.at(7) == 0){
        pass7 = true;
    }
    assert(pass7 == true);
    
    // case 4, Rover 2 to the west
    R2.x_position = -1;
    R2.y_position = 0;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 7 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) != 0){
        pass8 = true;
    }
    assert(pass8 == true);
    
    if(pass5 && pass6 && pass7 && pass8){
        passfail = true;
    }
    assert(passfail == true);
    return passfail;
}

void custom_test(){
    Rover R;
    POI P;
    R.x_position = 0;
    R.y_position = 0;
    R.theta = 90;
    
    P.x_position_poi = 0.56;
    P.y_position_poi = -1.91;
    P.value_poi = 100;
    
    R.reset_sensors();
    R.sense_poi(P.x_position_poi,P.y_position_poi,P.value_poi);
    
    
}

//x and y position of poi
vector< vector <double> > poi_positions;
vector<double> poi_positions_loc;

void stationary_rover_test(double x_start,double y_start){//Pass x_position,y_position
    Rover R_obj; //Rover object
    POI P_obj;
    
    R_obj.reset_sensors();
    
    //x and y position of poi
    vector< vector <double> > poi_positions;
    vector<double> poi_positions_loc;
    
    R_obj.x_position =x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    int radius = 2;
    
    double angle=0;
    
    P_obj.value_poi=100;
    
    int quad_0=0,quad_1=0,quad_2=0,quad_3=0,quad_0_1=0;
    while (angle<360) {
        if ((0<=angle && 45>= angle)) {
            quad_0++;
        }else if ((45<angle && 135>= angle)) {
            // do something in Q2
            quad_1++;
        }else if((135<angle && 225>= angle)){
            //do something in Q3
            quad_2++;
        }else if((225<angle && 315>= angle)){
            //do something in Q4
            quad_3++;
        }else if ((315<angle && 360> angle)){
            quad_0_1++;
        }
        poi_positions_loc.push_back(R_obj.x_position+(radius*cos(angle * (PI /180))));
        poi_positions_loc.push_back(R_obj.y_position+(radius*sin(angle * (PI /180))));
        poi_positions.push_back(poi_positions_loc);
        poi_positions_loc.clear();
        angle+=7;
    }
    
    vector<bool> checkPass_quad_1,checkPass_quad_2,checkPass_quad_3,checkPass_quad_0;
    
    for (int i=0; i<poi_positions.size(); i++) {
        for (int j=0; j<poi_positions.at(i).size(); j++) {
            P_obj.x_position_poi = poi_positions.at(i).at(j);
            P_obj.y_position_poi = poi_positions.at(i).at(++j);
            R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
            if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
                checkPass_quad_0.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0){
                checkPass_quad_1.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0){
                checkPass_quad_2.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0){
                checkPass_quad_3.push_back(true);
            }
            R_obj.reset_sensors();
        }
    }
    if (checkPass_quad_0.size() != (quad_0_1+quad_0)) {
        cout<<"Something wrong with quad_0"<<endl;;
    }else if (checkPass_quad_1.size() != (quad_1)){
        cout<<"Something wrong with quad_1"<<endl;
    }else if (checkPass_quad_2.size() != quad_2){
        cout<<"Something wrong with quad_2"<<endl;
    }else if (checkPass_quad_3.size() != quad_3){
        cout<<"Something wrong with quad_3"<<endl;
    }
}

void find_x_y_stationary_rover_test_1(double angle, double radius, double x_position, double y_position){
    poi_positions_loc.push_back(x_position+(radius*cos(angle * (PI /180))));
    poi_positions_loc.push_back(y_position+(radius*sin(angle * (PI /180))));
}

void stationary_rover_test_1(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj;
    
    R_obj.reset_sensors();
    
    R_obj.x_position =x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    int radius = 2;
    
    bool check_pass = false;
    
    double angle=0;
    
    P_obj.value_poi=100;
    
    while (angle<360) {
        find_x_y_stationary_rover_test_1(angle, radius, R_obj.x_position, R_obj.y_position);
        P_obj.x_position_poi = poi_positions_loc.at(0);
        P_obj.y_position_poi = poi_positions_loc.at(1);
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 0"<<endl;
            }
            check_pass = true;
        }else  if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 1"<<endl;
                
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 2"<<endl;
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 3"<<endl;
            }
            check_pass = true;
        }else{
            cout<<"Issue at an angle ::"<<angle<<" with x_position and y_position"<<R_obj.x_position<<R_obj.y_position<<endl;
            exit(10);
        }
        assert(check_pass==true);
        poi_positions_loc.clear();
        R_obj.reset_sensors();
        angle+=7;
        check_pass=false;
    }
}

void stationary_poi_test(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    vector<double> rover_position_loc;
    
    R_obj.reset_sensors();
    
    P_obj.x_position_poi=x_start;
    P_obj.y_position_poi=y_start;
    P_obj.value_poi=100;
    R_obj.theta=0.0;
    
    R_obj.x_position =0.0;
    R_obj.y_position =0.0;
    
    bool check_pass = false;
    
    for (int i=0; i<=R_obj.theta; ) {
        if (R_obj.theta > 360) {
            break;
        }
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        if (VERBOSE) {
            cout<<endl;
            for (int j=0; j<R_obj.sensors.size(); j++) {
                cout<<R_obj.sensors.at(j)<<"\t";
            }
            cout<<endl;
        }
        if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 0"<<endl;
            }
            check_pass = true;
        }else  if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 1";
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 2";
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 3";
            }
            check_pass = true;
        }else{
            cout<<"Issue at an angle ::"<<R_obj.theta<<" with x_position and y_position"<<P_obj.x_position_poi<<P_obj.y_position_poi<<endl;
            exit(10);
        }
        assert(check_pass==true);
        i+=7;
        R_obj.theta+=7;
        R_obj.reset_sensors();
    }
}

void two_rovers_test(double x_start, double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    vector<double> rover_position_loc;
    
    R_obj.reset_sensors();
    
    double otherRover_x = x_start;
    double otherRover_y = y_start;
    P_obj.value_poi=100;
    R_obj.theta=0.0;
    
    R_obj.x_position =0.0;
    R_obj.y_position =0.0;
    
    bool check_pass = false;
    
    for (int i=0; i<=R_obj.theta; ) {
        if (R_obj.theta > 360) {
            break;
        }
        R_obj.sense_rover(otherRover_x, otherRover_y);
        if (VERBOSE) {
            cout<<endl;
            for (int j=0; j<R_obj.sensors.size(); j++) {
                cout<<R_obj.sensors.at(j)<<"\t";
            }
            cout<<endl;
        }
        if (R_obj.sensors.at(4) != 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) == 0) {
            if ((0<=R_obj.theta && 45>= R_obj.theta)||(315<R_obj.theta && 360>= R_obj.theta)) {
                if (VERBOSE) {
                    cout<<"Pass Quad 0"<<endl;
                }
                check_pass = true;
            }
            
        }else  if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) != 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) == 0) {
            if((45<R_obj.theta && 135>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 1";
                }
                check_pass = true;
            }
        }else if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) !=0 && R_obj.sensors.at(7) == 0) {
            if((135<R_obj.theta && 225>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 2";
                }
                check_pass = true;
            }
        }else if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) != 0) {
            if((225<R_obj.theta && 315>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 3";
                }
                check_pass = true;
            }
        }else{
            cout<<"Issue at an angle ::"<<R_obj.theta<<" with x_position and y_position"<<P_obj.x_position_poi<<P_obj.y_position_poi<<endl;
            exit(10);
        }
        assert(check_pass==true);
        i+=7;
        R_obj.theta+=7;
        R_obj.reset_sensors();
    }
    
}

vector<double> row_values;
vector< vector <double> > assert_check_values;

void fill_assert_check_values(){
    //First set of x , y thetha values
    for(int i=0;i<3;i++)
        row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //second set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(1);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //third set of x,y,thetha values
    row_values.push_back(1);
    row_values.push_back(2);
    row_values.push_back(45);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //fourth set of x,y,thetha values
    row_values.push_back(1);
    row_values.push_back(3);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //fifth set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(4);
    row_values.push_back(315);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //sixth set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(5);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
}

bool tolerance(double delta_maniplate,double check_value){
    double delta = 0.0000001;
    if (((delta+ delta_maniplate)>check_value)|| ((delta- delta_maniplate)<check_value) || (( delta_maniplate)==check_value)) {
        return true;
    }else{
        return false;
    }
}


void test_path(double x_start, double y_start){
    bool VERBOSE = false;
    Rover R_obj;
    POI P_obj;
    
    //given
    R_obj.x_position=x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    
    P_obj.x_position_poi=1.0;
    P_obj.y_position_poi=1.0;
    P_obj.value_poi=100;
    
    
    
    fill_assert_check_values();
    
    int step_number = 0;
    bool check_assert = false;
    
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==0) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    double dx=0.0,dy=1.0;
    R_obj.move_rover(dx, dy);
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==1) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    
    dx=1.0;
    dy=1.0;
    R_obj.move_rover(dx, dy);
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==2) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=-1/sqrt(2.0);
    dy=1/sqrt(2.0);
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==3) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=-1.0;
    dy=1.0;
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==4) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=1/sqrt(2.0);
    dy=1/sqrt(2.0);
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==5) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
}

vector< vector <double> > point_x_y_circle;
vector<double> temp;

void find_x_y_test_circle_path(double start_x_position,double start_y_position,double angle){
    double radius = 1.0;
    temp.push_back(start_x_position+(radius*cos(angle * (PI /180))));
    temp.push_back(start_y_position+(radius*sin(angle * (PI/180))));
}

void test_circle_path(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj;
    POI P_obj;
    
    P_obj.x_position_poi=0.0;
    P_obj.y_position_poi=0.0;
    P_obj.value_poi=100.0;
    
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    
    double dx=0.0,dy=1.0;
    double angle=0.0;
    
    for(;angle<=360;){
        R_obj.x_position=x_start;
        R_obj.y_position=y_start;
        R_obj.theta=0.0;
        find_x_y_test_circle_path(x_start, y_start,angle);
        dx=temp.at(0);
        dy=temp.at(1);
        R_obj.move_rover(dx, dy);
        assert(tolerance(R_obj.x_position, dx));
        assert(tolerance(R_obj.y_position, dy));
        assert(tolerance(R_obj.theta, angle));
        temp.clear();
        angle+=15.0;
    }
    
}

void test_all_sensors(){
    POI_sensor_test();
    rover_sensor_test();
    custom_test();
    double x_start = 0.0, y_start = 0.0;
    stationary_rover_test(x_start,y_start);
    stationary_rover_test_1(x_start, y_start);
    stationary_poi_test(x_start,y_start);
    two_rovers_test(x_start,y_start);
    test_path(x_start,y_start);
    x_start = 0.0, y_start = 0.0;
    test_circle_path(x_start,y_start);
}



void repopulate(vector<Rover>* teamRover,int number_of_neural_network){
    for (int rover_number =0; rover_number < teamRover->size(); rover_number++) {
        //vector<unsigned> a;
        for (int neural_network =0; neural_network < (number_of_neural_network/2); neural_network++) {
            int R = rand()%teamRover->at(rover_number).network_for_agent.size();
            //Net N(a);
            //N = teamRover->at(rover_number).network_for_agent.at(R);
            //N.mutate();
            //teamRover->at(rover_number).network_for_agent.push_back(N);
            teamRover->at(rover_number).network_for_agent.push_back(teamRover->at(rover_number).network_for_agent.at(R));
            teamRover->at(rover_number).network_for_agent.back().mutate();
        }
        assert(teamRover->at(rover_number).network_for_agent.size() == number_of_neural_network);
    }
}

/*************************************************************************************
 This is to check for blockage collision. Checks if agent collides with obstacle
 ************************************************************************************/

bool checking_blockage(vector<double>* p_blocks_x, vector<double>* p_blocks_y, double blocking_radius, double x_position, double y_position){
    vector<double> distance;
    double temp_distance = 0.0;
    for (int loop_counter = 0 ; loop_counter < p_blocks_x->size(); loop_counter++) {
        double temp_x = p_blocks_x->at(loop_counter) - x_position;
        double temp_y = p_blocks_y->at(loop_counter) - y_position;
        temp_distance = sqrt((temp_x*temp_x)+(temp_y*temp_y));
        distance.push_back(temp_distance);
    }
    assert(p_blocks_x->size() == distance.size());
    vector<bool> check_distance;
    for (int loop_counter = 0 ; loop_counter < distance.size(); loop_counter++) {
        if (distance.at(loop_counter) <= blocking_radius) {
            return false;
        }
    }
    return true;
}


/*********************************************************************************
 calculates distance between two points
 Checks if values are with in tolerance. If with in returns true
 ********************************************************************************/
double cal_distance(double x1, double y1 , double x2,double y2){
    double final_value = sqrt(((x1-x2)*(x1-x2))+ ((y1-y2)*(y1-y2)));
    return final_value;
}

bool check_for_tolerance(double ref, double value){
    
    double high_tolerance = ref + 0.1;
    double lower_tolerance = ref -0.1;
    
    if ((high_tolerance >= value ) && (lower_tolerance <= value)) {
        return true;
    }
    return false;
}


/******************************************************************************
 Below are the functions for flocking
 1. Collision avoidance should make sure agents dont collide with each other and also making sure it does not collide with abstracle.
 2. Flocking will return true of false. I will just check flocking pattern of all agents and return ture or false.
 ****************************************************************************/

bool collision_avoidance(vector<Rover>* teamRover, vector<double>* p_vec_distance_between_agents, double agent_collision_radius){
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int other_rover = 0 ; other_rover < teamRover->size(); other_rover++) {
            double temp_distance;
            if (rover_number != other_rover) {
                temp_distance = cal_distance(teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position, teamRover->at(other_rover).x_position, teamRover->at(other_rover).y_position);
                if (temp_distance < agent_collision_radius) {
                    bool check_tolerance;
                    check_tolerance = check_for_tolerance(agent_collision_radius,temp_distance);
                    if (!check_tolerance) {
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

bool flocking(vector<Rover>* teamRover, vector<double>* p_vec_distance_between_agents, int leader_number){
    
    vector<double> current_distance;
    double distance;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        distance = cal_distance(teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position, teamRover->at(leader_number).x_position, teamRover->at(leader_number).y_position);
        current_distance.push_back(distance);
    }
    
    assert( current_distance.size() == p_vec_distance_between_agents->size());
    
    bool temp_check = false;
    vector<bool> flag_tolerance;
    for (int index = 0 ; index < current_distance.size(); index++) {
        temp_check = check_for_tolerance(p_vec_distance_between_agents->at(index), current_distance.at(index));
        if (!temp_check) {
            return false;
        }
    }
    return true;
}

void cal_velocity(vector<Rover>* teamRover){
    //Calculate velocity
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int x = 0 , y = 1; y < teamRover->at(rover_number).x_position_vec.size(); x++,y++) {
            double distance = cal_distance(teamRover->at(rover_number).x_position_vec.at(x), teamRover->at(rover_number).y_position_vec.at(x), teamRover->at(rover_number).x_position_vec.at(y), teamRover->at(rover_number).y_position_vec.at(y));
            teamRover->at(rover_number).velocity.push_back(distance);
        }
    }
}

/**********************************************************************************
 Leader does the simulation sensing and generating path. 
 Followers will check if they can move to new position if they can move they move to new position.
 If followers cannot move to new position then leader will generate new numbers.
 In logic first all rovers move to new position, when they find any agent in blocked area they all move back to previous step. This step was taken as it was computationally easy than calculating new positions and checking blocking.
 *********************************************************************************/

void simulation(vector<Rover>* teamRover, POI* individualPOI, double scaling_number, int blocking_radius, vector<double>* p_blocks_x, vector<double>* p_blocks_y,vector<double>* p_vec_distance_between_agents, double agent_collision_radius){
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
    //Find the leader index number
    int leader_index = 99999999;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        if(teamRover->at(rover_number).leader)
            leader_index = rover_number;
    }
    
    assert(leader_index <= teamRover->size());
    
    //Timestep to run simulation
    for (int time_step = 0 ; time_step < 5000 ; time_step++) {
        
        // Set X Y and theta to keep track of previous values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(rover_number).previous_x_position = teamRover->at(rover_number).x_position;
            teamRover->at(rover_number).previous_y_position = teamRover->at(rover_number).y_position;
            teamRover->at(rover_number).previous_theta = teamRover->at(rover_number).theta;
        }
        
        // reset and sense new values
        teamRover->at(leader_index).reset_sensors(); // Reset all sensors
        teamRover->at(leader_index).sense_all_values(individualPOI->x_position_poi_vec, individualPOI->y_position_poi_vec, individualPOI->value_poi_vec); // sense all values
        
        //Change of input values
        for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
            teamRover->at(leader_index).sensors.at(change_sensor_values) /= scaling_number;
        }
        
        //Neural network generating values
        teamRover->at(leader_index).network_for_agent.at(0).feedForward(teamRover->at(leader_index).sensors);
        for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
            assert(!isnan(teamRover->at(leader_index).sensors.at(change_sensor_values)));
        }
        
        double dx = teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.at(0);
        double dy = teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.at(1);
        teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.clear();
        
        //Move rovers
        assert(!isnan(dx));
        assert(!isnan(dy));
        
        for (int rover_number = 0 ; rover_number< teamRover->size(); rover_number++) {
            teamRover->at(rover_number).move_rover(dx, dy);
            teamRover->at(rover_number).x_position_vec.push_back(teamRover->at(rover_number).x_position);
            teamRover->at(rover_number).y_position_vec.push_back(teamRover->at(rover_number).y_position);
        }
        /*
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(rover_number).move_rover(dx, dy);
        }
        teamRover->at(leader_index).x_position_vec.push_back(teamRover->at(leader_index).x_position);
        teamRover->at(leader_index).y_position_vec.push_back(teamRover->at(leader_index).y_position);
        */
        
        //Check for blockage
        bool agent_on_block=false;
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            agent_on_block = checking_blockage(p_blocks_x, p_blocks_y, blocking_radius, teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position);
            if (!agent_on_block) {
                break;
            }
        }
        
        //check for flocking
        bool check_flocking = false;
        check_flocking = flocking(teamRover, p_vec_distance_between_agents,leader_index);
        
        //check for agent collision
        bool check_agent_collision = false;
        check_agent_collision = collision_avoidance(teamRover, p_vec_distance_between_agents, agent_collision_radius);
        
        if ((!agent_on_block) || (!check_flocking) || (!check_agent_collision)) {
            for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
                teamRover->at(rover_number).x_position = teamRover->at(rover_number).previous_x_position;
                teamRover->at(rover_number).y_position = teamRover->at(rover_number).previous_y_position;
                teamRover->at(rover_number).theta = teamRover->at(rover_number).previous_theta;
                time_step--;
                teamRover->at(rover_number).x_position_vec.pop_back();
                teamRover->at(rover_number).y_position_vec.pop_back();
            }
        }
    }
    
    cal_velocity(teamRover);
    
    // first select velocity
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).velocity_of_agent = rand()%2;
    }
    
    // flcoking and velocity
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        if ( rover_number != leader_index) {
            for (int time = 0 ; time < teamRover->at(leader_index).x_position_vec.size(); time++) {
                
            }
        }
    }
    
    FILE* p_xy;
    p_xy = fopen("XY_leader.txt", "a");
    for (int position = 0 ; position < teamRover->at(leader_index).x_position_vec.size(); position++) {
        fprintf(p_xy, "%f \t %f \n", teamRover->at(leader_index).x_position_vec.at(position), teamRover->at(leader_index).y_position_vec.at(position));
    }
    fclose(p_xy);
    
    FILE* p_xy_f;
    p_xy_f = fopen("XY_followers.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0 ; position < teamRover->at(rover_number).x_position_vec.size(); position++) {
            fprintf(p_xy_f, "%f \t %f \n", teamRover->at(rover_number).x_position_vec.at(position), teamRover->at(rover_number).y_position_vec.at(position));
        }
        fprintf(p_xy_f, "\n");
    }
    fclose(p_xy_f);
    
    FILE* p_velocity;
    p_velocity = fopen("Velocity.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0 ; position < teamRover->at(rover_number).velocity.size(); position++) {
            fprintf(p_velocity, "%f \t",teamRover->at(rover_number).velocity.at(position));
        }
        fprintf(p_velocity, "\n");
    }
    fclose(p_velocity);
}


void simulation_each_rover(vector<Rover>* teamRover, POI* individualPOI, double scaling_number, int blocking_radius, vector<double>* p_blocks_x, vector<double>* p_blocks_y, vector<double>* p_vec_distance_between_agents, double agent_collision_radius){
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
    //Find the leader index number
    int leader_index = 99999999;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        if(teamRover->at(rover_number).leader)
            leader_index = rover_number;
    }
    
    assert(leader_index <= teamRover->size());
    
    //Timestep to run simulation
    for (int time_step = 0 ; time_step < 5000 ; time_step++) {
        
        // Set X Y and theta to keep track of previous values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(rover_number).previous_x_position = teamRover->at(rover_number).x_position;
            teamRover->at(rover_number).previous_y_position = teamRover->at(rover_number).y_position;
            teamRover->at(rover_number).previous_theta = teamRover->at(rover_number).theta;
        }
        
        // reset and sense new values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(rover_number).reset_sensors(); // Reset all sensors
            teamRover->at(rover_number).sense_all_values(individualPOI->x_position_poi_vec, individualPOI->y_position_poi_vec, individualPOI->value_poi_vec); // sense all values
        }
        
        //Change of input values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
                teamRover->at(leader_index).sensors.at(change_sensor_values) /= scaling_number;
            }
        }
        
        //Neural network generating values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(leader_index).network_for_agent.at(0).feedForward(teamRover->at(leader_index).sensors);
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
                assert(!isnan(teamRover->at(leader_index).sensors.at(change_sensor_values)));
            }
            double dx = teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.at(0);
            double dy = teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.at(1);
            teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.clear();
            
            //Move rovers
            assert(!isnan(dx));
            assert(!isnan(dy));
            for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
                teamRover->at(rover_number).move_rover(dx, dy);
            }
            
            teamRover->at(rover_number).x_position_vec.push_back(teamRover->at(rover_number).x_position);
            teamRover->at(rover_number).y_position_vec.push_back(teamRover->at(rover_number).y_position);
        }
        
        //Check for blockage
        bool check_agent_on_block=false;
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            check_agent_on_block = checking_blockage(p_blocks_x, p_blocks_y, blocking_radius, teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position);
            if (!check_agent_on_block) {
                break;
            }
        }
        
        //check for flocking
        bool check_flocking = false;
        check_flocking = flocking(teamRover, p_vec_distance_between_agents,leader_index);
        
        //check for agent collision
        bool check_agent_collision = false;
        check_agent_collision = collision_avoidance(teamRover, p_vec_distance_between_agents, agent_collision_radius);
        
        if ((!check_agent_on_block) || (!check_flocking) || (!check_agent_collision)) {
            for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
                teamRover->at(rover_number).x_position = teamRover->at(rover_number).previous_x_position;
                teamRover->at(rover_number).y_position = teamRover->at(rover_number).previous_y_position;
                teamRover->at(rover_number).theta = teamRover->at(rover_number).previous_theta;
                time_step--;
                teamRover->at(rover_number).x_position_vec.pop_back();
                teamRover->at(rover_number).y_position_vec.pop_back();
            }
        }
    }
    
    cal_velocity(teamRover);
    
    
    FILE* p_xy;
    p_xy = fopen("XY_1.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0 ; position < teamRover->at(rover_number).x_position_vec.size(); position++) {
            fprintf(p_xy, "%f \t %f \n", teamRover->at(rover_number).x_position_vec.at(position), teamRover->at(rover_number).y_position_vec.at(position));
        }
        fprintf(p_xy, "\n");
    }
    fclose(p_xy);
    
    FILE* p_velocity;
    p_velocity = fopen("Velocity_1.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0 ; position < teamRover->at(rover_number).velocity.size(); position++) {
            fprintf(p_velocity, "%f \t",teamRover->at(rover_number).velocity.at(position));
        }
        fprintf(p_velocity, "\n");
    }
    fclose(p_velocity);
    
}


void simulation_new_try(vector<Rover>* teamRover, POI* individualPOI, double scaling_number, int blocking_radius, vector<double>* p_blocks_x, vector<double>* p_blocks_y, vector<double>* p_vec_distance_between_agents, double agent_collision_radius){
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
    //Find the leader index number
    int leader_index = 99999999;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        if(teamRover->at(rover_number).leader)
            leader_index = rover_number;
    }
    
    assert(leader_index <= teamRover->size());
    
    //Timestep to run simulation
    for (int time_step = 0 ; time_step < 5000 ; time_step++) {
        
        // Set X Y and theta to keep track of previous values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(rover_number).previous_x_position = teamRover->at(rover_number).x_position;
            teamRover->at(rover_number).previous_y_position = teamRover->at(rover_number).y_position;
            teamRover->at(rover_number).previous_theta = teamRover->at(rover_number).theta;
        }
        
        // reset and sense new values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(rover_number).reset_sensors(); // Reset all sensors
            teamRover->at(rover_number).sense_all_values(individualPOI->x_position_poi_vec, individualPOI->y_position_poi_vec, individualPOI->value_poi_vec); // sense all values
        }
        
        //Change of input values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
                teamRover->at(leader_index).sensors.at(change_sensor_values) /= scaling_number;
            }
        }
        
        //Neural network generating values
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            teamRover->at(leader_index).network_for_agent.at(0).feedForward(teamRover->at(leader_index).sensors);
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
                assert(!isnan(teamRover->at(leader_index).sensors.at(change_sensor_values)));
            }
            double dx = teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.at(0);
            double dy = teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.at(1);
            teamRover->at(leader_index).network_for_agent.at(0).outputvaluesNN.clear();
            
            //Move rovers
            assert(!isnan(dx));
            assert(!isnan(dy));
            for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
                teamRover->at(rover_number).move_rover(dx, dy);
            }
            
            teamRover->at(rover_number).x_position_vec.push_back(teamRover->at(rover_number).x_position);
            teamRover->at(rover_number).y_position_vec.push_back(teamRover->at(rover_number).y_position);
        }
        
        //Check for blockage
        /*
        bool check_agent_on_block=false;
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            check_agent_on_block = checking_blockage(p_blocks_x, p_blocks_y, blocking_radius, teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position);
            if (!check_agent_on_block) {
                break;
            }
        }
        
        //check for flocking
        //bool check_flocking = false;
        //check_flocking = flocking(teamRover, p_vec_distance_between_agents,leader_index);
        
        //check for agent collision
        //bool check_agent_collision = false;
        //check_agent_collision = collision_avoidance(teamRover, p_vec_distance_between_agents, agent_collision_radius);
        /*
        if ((!check_agent_on_block)) {
            for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
                teamRover->at(rover_number).x_position = teamRover->at(rover_number).previous_x_position;
                teamRover->at(rover_number).y_position = teamRover->at(rover_number).previous_y_position;
                teamRover->at(rover_number).theta = teamRover->at(rover_number).previous_theta;
                time_step--;
                teamRover->at(rover_number).x_position_vec.pop_back();
                teamRover->at(rover_number).y_position_vec.pop_back();
            }
        }
         */
    }
    
    cal_velocity(teamRover);
    
    
    FILE* p_xy;
    p_xy = fopen("XY_1.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0 ; position < teamRover->at(rover_number).x_position_vec.size(); position++) {
            fprintf(p_xy, "%f \t %f \n", teamRover->at(rover_number).x_position_vec.at(position), teamRover->at(rover_number).y_position_vec.at(position));
        }
        fprintf(p_xy, "\n");
    }
    fclose(p_xy);
    
    FILE* p_velocity;
    p_velocity = fopen("Velocity_1.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0 ; position < teamRover->at(rover_number).velocity.size(); position++) {
            fprintf(p_velocity, "%f \t",teamRover->at(rover_number).velocity.at(position));
        }
        fprintf(p_velocity, "\n");
    }
    fclose(p_velocity);
    
    /***************************
     Punishment applied
     **************************/
    //This is to check collision avodiance with obstacle
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int position = 0  ; position < teamRover->at(rover_number).x_position_vec.size(); position++) {
            bool check_blockage;
            check_blockage = checking_blockage(p_blocks_x, p_blocks_y, blocking_radius, teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position);
            if (!check_blockage) {
                teamRover->at(rover_number).blockage = 1000000;
                break;
            }
        }
    }
    
    //Check for flocking
    vector<double> current_distance;
    double distance;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        distance = cal_distance(teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position, teamRover->at(leader_index).x_position, teamRover->at(leader_index).y_position);
        current_distance.push_back(distance);
    }
    
    assert( current_distance.size() == p_vec_distance_between_agents->size());
    
    bool temp_check = false;
    vector<bool> flag_tolerance;
    for (int index = 0 ; index < current_distance.size(); index++) {
        temp_check = check_for_tolerance(p_vec_distance_between_agents->at(index), current_distance.at(index));
        if (!temp_check) {
            
        }
    }


}



/******************************************************************************************
 Dynamic simulation functions
 *****************************************************************************************/

//Calculate zigma_norm
double zigma_norm(vector<double>* p_values){
    double sumation = 0;
    for (int size=0; size < p_values->size(); size++) {
        sumation += pow(p_values->at(size), 2);
    }
    
    return sqrt(sumation);
}

double zigma_1(vector<double>* p_values){
    double norm_value = zigma_norm(p_values);
    return norm_value/(pow(sqrt(1+norm_value),2));
}

// Calculate n_i
vector<double> n_i(int agent_number, vector<Rover>* p_rover){
    vector<double> values;
    vector<double>* p_values = &values;
    
    //Calculate r_alpha value
    p_values->push_back(r);
    double r_alpha = zigma_norm(p_values);
    p_values->clear();
    assert(p_values->size() == 0 );
    
    int k = 0;
    
    vector<double> zero_vector;
    for (int rover = 0 ; rover < p_rover->size(); rover++) {
        zero_vector.push_back(0);
    }
    
    //Save the agent number
    vector<double> temp_saved_number;
    
    
    for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
        if (agent_number != rover_number) {
            //calculate temp_r_alpha
            p_values->push_back(p_rover->at(rover_number).x_position - p_rover->at(agent_number).x_position);
            p_values->push_back(p_rover->at(rover_number).y_position - p_rover->at(agent_number).y_position);
            double temp_r_alpha = zigma_norm(p_values);
            p_values->clear();
            assert(p_values->size() == 0);
            
            if (temp_r_alpha <= r_alpha) {
                k++;
                zero_vector.at(k) = rover_number;
                temp_saved_number.push_back(rover_number);
            }
        }
    }
    
    return temp_saved_number;
    
}

double ro_h(double z, double h){
    if (z>=0 && z<h) {
        return 1;
    }else if (z>=h && z<=1){
        return 0.5*(1+cos(PI*((z-h)/(1-h))));
    }
    return 0;
}

double zigma_1(double z){
    return z/(sqrt(1+ pow(norm(z),2)));
}

//aa bb
double phi(double z){
    double c = abs(aa-bb)/sqrt(4*aa*bb);
    return (1/2)*((aa+bb)*zigma_1(z+c)+(aa-bb));
}

double phi_alpha(double z,double r_alpha,double d_alpha){
    return ro_h(z,h_a)*phi((z-d_alpha));
}

vector<double> nij(vector<double> qi, vector<double> qj){
    vector<double> return_vec;
    assert(qi.size() == qj.size());
    for (int index = 0 ; index < qi.size(); index++) {
        double temp = qj.at(index) - qi.at(index);
        double temp_1 = (sqrt(1+epsilon*((pow(norm(qj.at(index)-qi.at(index)),2)))));
        return_vec.push_back(temp/temp_1);
    }
    return return_vec;
}

void fi_alpha(int agent_number,vector<Rover>* p_rover){
    vector<double> values;
    vector<double>* p_values = &values;
    p_values->push_back(r);
    double r_alpha = zigma_norm(p_values);
    p_values->clear();
    p_values->push_back(d);
    double d_alpha = zigma_norm(p_values);
    p_values->clear();
    
    assert(p_values->size() == 0);
    
    vector<vector<double>> agent_numbers;
    
    agent_numbers.push_back(n_i(agent_number, p_rover));
    
    double sum_1 = 0;
    double sum_2 = 0;
    
    for (int index_number = 0 ; index_number < agent_numbers.at(0).size(); index_number++) {
        for (int index_number_1 = 0; index_number_1 < p_rover->size(); index_number_1++) {
            if (agent_numbers.at(0).at(index_number) == index_number_1) {
                p_values->push_back(p_rover->at(agent_number).x_position - p_rover->at(index_number_1).x_position);
                p_values->push_back(p_rover->at(agent_number).y_position - p_rover->at(index_number_1).y_position);
                
                
                sum_1 += phi_alpha(zigma_1(p_values), r_alpha, d_alpha);
                
                p_values->clear();
            }
        }
    }
}


void fi_gamma(int agent_number, vector<Rover>* p_rover){
    vector<double> value;
    vector<double>* p_values = &value;
    value.push_back(p_rover->at(agent_number).x_position - qd.at(0));
    double temp_1 = zigma_1(p_values);
    value.clear();
    value.push_back(p_rover->at(agent_number).y_position - qd.at(1));
    double temp_2 = zigma_1(p_values);
    value.clear();
    
    double temp_3 = 0;
    double temp_4 = 0;
    
    
    double first_value = -c1_g*(temp_1)-c2_g;
    
}

void fi_beta(int agent_number, vector<Rover>* p_rover){
    
}

void dynamic_simulation_run(vector<Rover>* teamRover, POI* individualPOI, double scaling_number, int blocking_radius, vector<double>* p_blocks_x, vector<double>* p_blocks_y, double agent_collision_radius, vector<unsigned>* p_topology,vector<double>* p_t){
    
    vector<vector<double>> agent_numbers;
    
    for (int time_step = 0; time_step < p_t->size(); time_step++) {
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            
            agent_numbers.push_back(n_i(rover_number, teamRover));
            fi_alpha(rover_number,teamRover);
            fi_gamma(rover_number, teamRover);
        }
    }
    
}



void dynamic_simulation(vector<Rover>* teamRover, POI* individualPOI, double scaling_number, int blocking_radius, vector<double>* p_blocks_x, vector<double>* p_blocks_y, vector<double>* p_vec_distance_between_agents, double agent_collision_radius, vector<unsigned>* p_topology,vector<double>* p_t){
    
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
    //Find the leader index number
    int leader_index = 99999999;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        if(teamRover->at(rover_number).leader)
            leader_index = rover_number;
    }
    
    assert(leader_index <= teamRover->size());
    
    int path_net = 10;
    
    teamRover->at(leader_index).create_path_network(path_net, p_topology);
    
    double save_x_position = teamRover->at(leader_index).x_position_vec.at(0);
    double save_y_position = teamRover->at(leader_index).y_position_vec.at(0);
    
    vector< vector <double> > x_values;
    vector< vector <double> > y_values;
    vector<double> best_path_x;
    vector<double> best_path_y;
    
    double best_path_index;
    
    int generations = 1;
    int number_of_steps = 250;
    
    
    for (int iterations = 0 ; iterations < generations; iterations++) {
        
        // Reset leader values;
        for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
            teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
            teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
            teamRover->at(temp_rover_number).theta = 0.0;
        }
        
        //Create 10 Neural Network
        for (int neural_network = 0 ; neural_network < teamRover->at(leader_index).path_finder_network.size();neural_network++) {
            //Set near distance to POI high
            teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance = 9999999.9999;
            double lowest_distance = 9999999.9999;
            teamRover->at(leader_index).x_position  = teamRover->at(leader_index).x_position_vec.at(0);
            teamRover->at(leader_index).y_position = teamRover->at(leader_index).y_position_vec.at(0);
            teamRover->at(leader_index).theta = 0.0;
            
            //Timestep to run simulation
            for (int time_step = 0 ; time_step < number_of_steps ; time_step++) {
                
                // Set X Y and theta to keep track of previous values
                teamRover->at(leader_index).previous_x_position = teamRover->at(leader_index).x_position;
                teamRover->at(leader_index).previous_y_position = teamRover->at(leader_index).y_position;
                teamRover->at(leader_index).previous_theta = teamRover->at(leader_index).theta;
                
                // reset and sense new values
                teamRover->at(leader_index).reset_sensors(); // Reset all sensors
                teamRover->at(leader_index).sense_all_values(individualPOI->x_position_poi_vec, individualPOI->y_position_poi_vec, individualPOI->value_poi_vec); // sense all values
                
                //Change of input values
                for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
                    teamRover->at(leader_index).sensors.at(change_sensor_values) /= scaling_number;
                }
                
                //Neural network generating values
                teamRover->at(leader_index).path_finder_network.at(neural_network).feedForward(teamRover->at(leader_index).sensors);
                for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(leader_index).sensors.size(); change_sensor_values++) {
                    assert(!isnan(teamRover->at(leader_index).sensors.at(change_sensor_values)));
                }
                
                double dx = teamRover->at(leader_index).path_finder_network.at(neural_network).outputvaluesNN.at(0);
                double dy = teamRover->at(leader_index).path_finder_network.at(neural_network).outputvaluesNN.at(1);
                teamRover->at(leader_index).path_finder_network.at(neural_network).outputvaluesNN.clear();
                
                //Move rovers
                assert(!isnan(dx));
                assert(!isnan(dy));
                
                teamRover->at(leader_index).move_rover(dx, dy);
                teamRover->at(leader_index).x_position_vec.push_back(teamRover->at(leader_index).x_position);
                teamRover->at(leader_index).y_position_vec.push_back(teamRover->at(leader_index).y_position);
                
                
                //Check for blockage
                bool agent_on_block=false;
                for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
                    agent_on_block = checking_blockage(p_blocks_x, p_blocks_y, blocking_radius, teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position);
                    if (!agent_on_block) {
                        teamRover->at(leader_index).path_finder_network.at(leader_index).fitness += -9999999;
                    }
                }
            }
            
            
            
            //Fnd the closest distance and fitness value
            for (int position = 0 ; position < teamRover->at(leader_index).x_position_vec.size(); position++) {
                double x_value = teamRover->at(leader_index).x_position_vec.at(position) - individualPOI->x_position_poi_vec.at(0);
                double y_value = teamRover->at(leader_index).y_position_vec.at(position) - individualPOI->y_position_poi_vec.at(0);
                double distance = sqrt((x_value*x_value)+(y_value*y_value));
                if (lowest_distance > distance) {
                    lowest_distance = distance;
                }
                teamRover->at(leader_index).path_finder_network.at(neural_network).distance_values.push_back(distance);
            }
            double index = *min_element(teamRover->at(leader_index).path_finder_network.at(neural_network).distance_values.begin(),teamRover->at(leader_index).path_finder_network.at(neural_network).distance_values.end());
            
            teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance = teamRover->at(leader_index).path_finder_network.at(neural_network).distance_values.at(index) ;
            
            assert(lowest_distance = teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance );
            
            if (teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance < 1) {
                teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance = 1;
            }
            
            teamRover->at(leader_index).path_finder_network.at(neural_network).fitness += (individualPOI->value_poi_vec.at(0)/teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance);
            
            x_values.push_back(teamRover->at(leader_index).x_position_vec);
            y_values.push_back(teamRover->at(leader_index).y_position_vec);
            
            if (iterations == generations-1) {
                FILE* p_file;
                p_file = fopen("XY_leader.txt", "a");
                for (int index = 0 ; index < teamRover->at(leader_index).x_position_vec.size(); index++) {
                    fprintf(p_file, "%f \t %f \n",teamRover->at(leader_index).x_position_vec.at(index),teamRover->at(leader_index).y_position_vec.at(index));
                }
                fprintf(p_file, "\n");
                fclose(p_file);
            }
            
            FILE* p_values;
            p_values = fopen("Values.txt", "a");
            fprintf(p_values, "%f \t %f \n",teamRover->at(leader_index).path_finder_network.at(neural_network).best_distance,teamRover->at(leader_index).path_finder_network.at(neural_network).fitness);
            fclose(p_values);
            
            
            teamRover->at(leader_index).x_position_vec.clear();
            teamRover->at(leader_index).y_position_vec.clear();
            teamRover->at(leader_index).path_finder_network.at(neural_network).distance_values.clear();
            teamRover->at(leader_index).x_position_vec.push_back(save_x_position);
            teamRover->at(leader_index).y_position_vec.push_back(save_y_position);
        }
        
        //EA
        for (int count = 0 ; count < (path_net/2); count++) {
            //Generate random numbers
            int random_number_1 = rand()%teamRover->at(leader_index).path_finder_network.size();
            int random_number_2 = rand()%teamRover->at(leader_index).path_finder_network.size();
            while ((random_number_1 == random_number_2) || (random_number_1 == teamRover->at(leader_index).path_finder_network.size()) || (random_number_2 == teamRover->at(leader_index).path_finder_network.size())) {
                random_number_2 = rand()%teamRover->at(leader_index).path_finder_network.size();
                random_number_1 = rand()%teamRover->at(leader_index).path_finder_network.size();
            }
            
            double fitness_1 = teamRover->at(leader_index).path_finder_network.at(random_number_1).fitness;
            double fitness_2 = teamRover->at(leader_index).path_finder_network.at(random_number_2).fitness;
            
            if (fitness_1 < fitness_2) {
                teamRover->at(leader_index).path_finder_network.erase(teamRover->at(leader_index).path_finder_network.begin()+random_number_2);
            }else{
                teamRover->at(leader_index).path_finder_network.erase(teamRover->at(leader_index).path_finder_network.begin()+random_number_1);
            }
        }
        
        assert(teamRover->at(leader_index).path_finder_network.size() == path_net/2);
        
        for (int neural_network =0; neural_network < (path_net/2); neural_network++) {
            int R = rand()%teamRover->at(leader_index).path_finder_network.size();
            teamRover->at(leader_index).path_finder_network.push_back(teamRover->at(leader_index).path_finder_network.at(R));
            teamRover->at(leader_index).path_finder_network.back().mutate();
        }
        
        assert(teamRover->at(leader_index).path_finder_network.size() == path_net);
        
        if (iterations == generations-1) {
            best_path_index = 9999999;
            double temp_best = teamRover->at(leader_index).path_finder_network.at(0).fitness;
            for (int net = 0 ; net < teamRover->at(leader_index).path_finder_network.size(); net++) {
                if (temp_best > teamRover->at(leader_index).path_finder_network.at(net).fitness) {
                    temp_best = teamRover->at(leader_index).path_finder_network.at(net).fitness;
                    best_path_index = net;
                }
            }
            
            for (int index = 0 ; index < x_values.at(best_path_index).size(); index++) {
                best_path_x.push_back(x_values.at(best_path_index).at(index));
                best_path_y.push_back(y_values.at(best_path_index).at(index));
            }
        }
        
        if (iterations != generations-1) {
            x_values.clear();
            y_values.clear();
        }
    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).x_position = teamRover->at(rover_number).x_position_vec.at(0);
        teamRover->at(rover_number).y_position = teamRover->at(rover_number).y_position_vec.at(0);
        teamRover->at(rover_number).theta = 0.0;
        teamRover->at(rover_number).velocity_of_agent_x = ((double)rand()) / ((double)RAND_MAX) * 1.0 + 0.0;// make it vector with x and y position
        teamRover->at(rover_number).velocity_of_agent_y = ((double)rand()) / ((double)RAND_MAX) * 1.0 + 0.0;
    }
    
    //Place each rover at random location
    
    
    
    for (int time_step = 0 ; time_step < number_of_steps ; time_step++) {
        //calculate path for each agent
        
    }
    
    cout<<"Done"<<endl;
}

/***************************
 Main
 **************************/

int main(int argc, const char * argv[]) {
     srand((unsigned)time(NULL));
    srand(time(NULL));
    if (test_simulation) {
        test_all_sensors();
        cout<<"All Test"<<endl;
    }
    
    //    for (int stat_run =0 ; stat_run < 30; stat_run++) {
    if (run_simulation) {

        //First set up environment
        int number_of_rovers = 150;
        
        
        //Set values of poi's
        POI individualPOI;
        POI* p_poi = &individualPOI;
        
        //Create POI
        individualPOI.x_position_poi_vec.push_back(50.0);
        individualPOI.y_position_poi_vec.push_back(50.0);
        individualPOI.value_poi_vec.push_back(100000);
        
        //vectors of rovers
        vector<Rover> teamRover;
        vector<Rover>* p_rover = &teamRover;
        Rover a;
        for (int i=0; i<number_of_rovers; i++) {
            teamRover.push_back(a);
        }
        /*
        for (int i=0 ; i<number_of_rovers; i++) {
            teamRover.at(i).x_position_vec.push_back(0+(0.5*i));
            teamRover.at(i).y_position_vec.push_back(0);
        }
        */
        
        for (int rover_number = 0; rover_number <teamRover.size(); rover_number++) {
            double x=(double)rand()/(RAND_MAX + 1)+1+(rand()%50);
            double y=(double)rand()/(RAND_MAX + 1)+1+(rand()%50);
            teamRover.at(rover_number).x_position_vec.push_back(x);
            teamRover.at(rover_number).y_position_vec.push_back(y);
            teamRover.at(rover_number).x_position = x;
            teamRover.at(rover_number).y_position = y;
        }
        //Second set up neural networks
        //Create numNN of neural network with pointer
        int numNN = 1;
        vector<unsigned> topology;
        vector<unsigned>* p_topology = &topology;
        topology.clear();
        topology.push_back(8);
        topology.push_back(14);
        topology.push_back(2);
        
        for (int rover_number =0 ; rover_number < number_of_rovers; rover_number++) {
            teamRover.at(rover_number).create_neural_network_population(numNN, topology);
        }
        
        //setting all rovers to inital state
        for (int temp_rover_number =0 ; temp_rover_number<teamRover.size(); temp_rover_number++) {
            teamRover.at(temp_rover_number).x_position = teamRover.at(temp_rover_number).x_position_vec.at(0);
            teamRover.at(temp_rover_number).y_position = teamRover.at(temp_rover_number).y_position_vec.at(0);
            teamRover.at(temp_rover_number).previous_x_position = teamRover.at(temp_rover_number).x_position_vec.at(0);
            teamRover.at(temp_rover_number).previous_y_position = teamRover.at(temp_rover_number).y_position_vec.at(0);
            teamRover.at(temp_rover_number).theta = 0.0;
            teamRover.at(temp_rover_number).previous_theta = 0.0;
            
        }
        
        bool b_location_agents = false;
        
        if (b_location_agents) {
            FILE* p_agent;
            p_agent = fopen("location.txt", "a");
            for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
                fprintf(p_agent, "%f \t %f \n", teamRover.at(rover_number).x_position, teamRover.at(rover_number).y_position);
            }
            fclose(p_agent);
            
        }
        
        
        //Find Scaling Number
        double scaling_number = find_scaling_number(p_rover,p_poi);
        
        //Blocking Area
        double radius_blocking = 20.0;
        vector<double> blocks_x;
        vector<double>* p_blocks_x = &blocks_x;
        vector<double> blocks_y;
        vector<double>* p_blocks_y = &blocks_y;
        double blocking_x_position = 25.0;
        double blocking_y_position = 25.0;
        
        blocks_x.push_back(blocking_x_position);
        blocks_y.push_back(blocking_y_position);
        
        assert(blocks_x.size() == blocks_y.size());
        
        //Select leader
        int select_leader = rand()%number_of_rovers;
        teamRover.at(select_leader).leader = true;
        
        double agent_collision_radius = 0.15;
        
        
        /*double distance;
        for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
            distance = cal_distance(teamRover.at(rover_number).x_position, teamRover.at(rover_number).y_position,teamRover.at(select_leader).x_position,teamRover.at(select_leader).y_position);
            vec_distance_between_agents.push_back(distance);
        }*/
        
        for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
            for (int rover_number_1 = 0 ; rover_number_1 <teamRover.size(); rover_number_1++) {
                double distance;
                if (rover_number_1 != rover_number) {
                    distance = cal_distance(teamRover.at(rover_number).x_position, teamRover.at(rover_number).y_position, teamRover.at(rover_number_1).x_position, teamRover.at(rover_number_1).y_position);
                    teamRover.at(rover_number).vec_distance_between_agents.push_back(distance);
                }
            }
         assert(teamRover.at(rover_number).vec_distance_between_agents.size() == number_of_rovers-1);
        }
        
        
        //bool test = checking_blockage(p_blocks_x, p_blocks_y, radius_blocking, teamRover.at(0).x_position_vec.at(0), teamRover.at(0).y_position_vec.at(0));
        
       /*
        for (int generation = 0 ; generation < 1 ; generation ++) {
            simulation(p_rover, p_poi, scaling_number, radius_blocking, p_blocks_x, p_blocks_y,p_vec_distance_between_agents, agent_collision_radius);
            //simulation_each_rover(p_rover, p_poi, scaling_number, radius_blocking, p_blocks_x, p_blocks_y, p_vec_distance_between_agents, agent_collision_radius);
        }
        
        //for (int generation = 0 ; generation < 10 ; generation ++) {
            //simulation_new_try(p_rover, p_poi, scaling_number, radius_blocking, p_blocks_x, p_blocks_y, p_vec_distance_between_agents, agent_collision_radius);
            
        //}
        
        */
        
        vector<double> t;
        vector<double>* p_t = &t;
        
        for (double count = 0.00; count < 70; ) {
            t.push_back(count);
            count = count + 0.09;
        }
        
        dynamic_simulation_run(p_rover, p_poi, scaling_number, radius_blocking, p_blocks_x, p_blocks_y,  agent_collision_radius, p_topology,p_t);
        
        
    }
    return 0;
}

