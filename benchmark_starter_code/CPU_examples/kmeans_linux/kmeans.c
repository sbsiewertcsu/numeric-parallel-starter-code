// kmeans.c
// Ethan Brodsky
// October 2011
//
// Adapted for use on Linux and with Quark by Sam Siewert, April 2016
//
// Test case using the sunset thumbnail added as demonstration.

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "image.h"

#define sqr(x) ((x)*(x))

#define MAX_CLUSTERS 4

#define MAX_ITERATIONS 10

#define BIG_double (INFINITY)

void fail(char *str)
{
    printf("%s", str);
    exit(-1);
}
  
double calc_distance(int dim, double *p1, double *p2)
{
    double distance_sq_sum = 0;
    int idx;
    
    for (idx = 0; idx < dim; idx++)
      distance_sq_sum += sqr(p1[idx] - p2[idx]);

    return distance_sq_sum;
    
}

void calc_all_distances(int dim, int n, int k, double *X, double *centroid, double *distance_output)
{
    int idx, jdx;

    for (idx = 0; idx < n; idx++) // for each point
      for (jdx = 0; jdx < k; jdx++) // for each cluster
      {
         // calculate distance between point and cluster centroid
         distance_output[idx*k + jdx] = calc_distance(dim, &X[idx*dim], &centroid[jdx*dim]);
      }
}
  
double calc_total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index)
// NOTE: a point with cluster assignment -1 is ignored
{
    double tot_D = 0;
    int idx;
    
   // for every point
    for (idx = 0; idx < n; idx++)
    {
       // which cluster is it in?
        int active_cluster = cluster_assignment_index[idx];
        
       // sum distance
        if (active_cluster != -1)
          tot_D += calc_distance(dim, &X[idx*dim], &centroids[active_cluster*dim]);
    }
      
    return tot_D;
}

void choose_all_clusters_from_distances(int dim, int n, int k, double *distance_array, int *cluster_assignment_index)
  {
   // for each point
    int idx, jdx;

    for (idx = 0; idx < n; idx++)
    {
        int best_index = -1;
        double closest_distance = BIG_double;
        
       // for each cluster
        for (jdx = 0; jdx < k; jdx++)
        {
           // distance between point and cluster centroid
           
            double cur_distance = distance_array[idx*k + jdx];
            if (cur_distance < closest_distance)
              {
                best_index = jdx;
                closest_distance = cur_distance;
              }
        }

       // record in array
        cluster_assignment_index[idx] = best_index;
    }
}

void calc_cluster_centroids(int dim, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid)
  {
    int cluster_member_count[MAX_CLUSTERS];
    int idx, jdx;
  
   // initialize cluster centroid coordinate sums to zero
    for (idx = 0; idx < k; idx++) 
    {
        cluster_member_count[idx] = 0;
        
        for (jdx = 0; jdx < dim; jdx++)
          new_cluster_centroid[idx*dim + jdx] = 0;
   }

   // sum all points
   // for every point
    for (idx = 0; idx < n; idx++)
    {
       // which cluster is it in?
        int active_cluster = cluster_assignment_index[idx];

       // update count of members in that cluster
        cluster_member_count[active_cluster]++;
        
       // sum point coordinates for finding centroid
        for (jdx = 0; jdx < dim; jdx++)
          new_cluster_centroid[active_cluster*dim + jdx] += X[idx*dim + jdx];
    }
     
      
   // now divide each coordinate sum by number of members to find mean/centroid
   // for each cluster
    for (idx = 0; idx < k; idx++) 
    {
        if (cluster_member_count[idx] == 0)
          printf("WARNING: Empty cluster %d! \n", idx);
          
       // for each dimension
        for (jdx = 0; jdx < dim; jdx++)
          new_cluster_centroid[idx*dim + jdx] /= cluster_member_count[idx];  /// XXXX will divide by zero here for any empty clusters!

    }
}

void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count)
{
   // initialize cluster member counts
    int idx;

    for (idx = 0; idx < k; idx++) 
      cluster_member_count[idx] = 0;
  
   // count members of each cluster    
    for (idx = 0; idx < n; idx++)
      cluster_member_count[cluster_assignment_index[idx]]++;
}

void update_delta_score_table(int dim, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count, double *point_move_score_table, int cc)
{
   // for every point (both in and not in the cluster)
    int idx, kdx;

    for (idx = 0; idx < n; idx++)
    {
        double dist_sum = 0;
        for (kdx = 0; kdx < dim; kdx++)
        {
            double axis_dist = X[idx*dim + kdx] - cluster_centroid[cc*dim + kdx]; 
            dist_sum += sqr(axis_dist);
        }
          
        double mult = ((double)cluster_member_count[cc] / (cluster_member_count[cc] + ((cluster_assignment_cur[idx]==cc) ? -1 : +1)));

        point_move_score_table[idx*dim + cc] = dist_sum * mult;
    }
}
  
  
void  perform_move(int dim, int n, int k, double *X, int *cluster_assignment, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster)
{
    int idx;

    int cluster_old = cluster_assignment[move_point];
    int cluster_new = move_target_cluster;
  
   // update cluster assignment array
    cluster_assignment[move_point] = cluster_new;
    
   // update cluster count array
    cluster_member_count[cluster_old]--;
    cluster_member_count[cluster_new]++;
    
    if (cluster_member_count[cluster_old] <= 1)
      printf("WARNING: Can't handle single-member clusters! \n");
    
   // update centroid array
    for (idx = 0; idx < dim; idx++)
    {
        cluster_centroid[cluster_old*dim + idx] -= (X[move_point*dim + idx] - cluster_centroid[cluster_old*dim + idx]) / cluster_member_count[cluster_old];
        cluster_centroid[cluster_new*dim + idx] += (X[move_point*dim + idx] - cluster_centroid[cluster_new*dim + idx]) / cluster_member_count[cluster_new];
    }
}  
  
void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid)
{
    int cluster_member_count[MAX_CLUSTERS];
    int idx;
    
    get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
     
    printf("  Final clusters \n");
    for (idx = 0; idx < k; idx++) 
      printf("    cluster %d:     members: %8d, centroid (%.1f %.1f) \n", idx, cluster_member_count[idx], cluster_centroid[idx*dim + 0], cluster_centroid[idx*dim + 1]);
}

void copy_assignment_array(int n, int *src, int *tgt)
{
    int idx;

    for (idx = 0; idx < n; idx++)
      tgt[idx] = src[idx];
}
  
int assignment_change_count(int n, int a[], int b[])
{
    int change_count = 0;
    int idx;

    for (idx = 0; idx < n; idx++)
      if (a[idx] != b[idx])
        change_count++;
        
    return change_count;
}

void kmeans(
            int  dim,		             // dimension of data 
            double *X,                       // pointer to data
            int   n,                         // number of elements
            int   k,                         // number of clusters
            double *cluster_centroid,        // initial cluster centroids
            int   *cluster_assignment_final  // output
           )
  {
    int idx;
    double *dist                    = (double *)malloc(sizeof(double) * n * k);
    int   *cluster_assignment_cur  = (int *)malloc(sizeof(int) * n);
    int   *cluster_assignment_prev = (int *)malloc(sizeof(int) * n);
    double *point_move_score        = (double *)malloc(sizeof(double) * n * k);
    int make_move = 0;
    int point_to_move = -1;
    int target_cluster = -1;
    int cluster_member_count[MAX_CLUSTERS];
    int online_iteration = 0;
    int last_point_moved = 0;
    
    
    if (!dist || !cluster_assignment_cur || !cluster_assignment_prev || !point_move_score)
      fail("Error allocating dist arrays");
    
   // initial setup  
    calc_all_distances(dim, n, k, X, cluster_centroid, dist);
    choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
    copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

   // BATCH UPDATE
    double prev_totD = BIG_double;
    int batch_iteration = 0;
    while (batch_iteration < MAX_ITERATIONS)
    {
        // printf("batch iteration %d \n", batch_iteration);
        // cluster_diag(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
        
        // update cluster centroids
        calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);

        // deal with empty clusters
        // XXXXXXXXXXXXXX

        // see if we've failed to improve
         double totD = calc_total_distance(dim, n, k, X, cluster_centroid, cluster_assignment_cur);
         if (totD > prev_totD)
          // failed to improve - currently solution worse than previous
           {
            // restore old assignments
             copy_assignment_array(n, cluster_assignment_prev, cluster_assignment_cur);
             
            // recalc centroids
             calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
             
             printf("  negative progress made on this step - iteration completed (%.2f) \n", totD - prev_totD);
             
            // done with this phase
             break;
           }
           
        // save previous step
         copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);
         
        // move all points to nearest cluster
         calc_all_distances(dim, n, k, X, cluster_centroid, dist);
         choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
         
         int change_count = assignment_change_count(n, cluster_assignment_cur, cluster_assignment_prev);
         
         printf("%3d   %u   %9d  %16.2f %17.2f\n", batch_iteration, 1, change_count, totD, totD - prev_totD);
         fflush(stdout);
         
        // done with this phase if nothing has changed
         if (change_count == 0)
           {
             printf("  no change made on this step - iteration completed \n");
             break;
           }

         prev_totD = totD;
                        
         batch_iteration++;
      }

    cluster_diag(dim, n, k, X, cluster_assignment_cur, cluster_centroid);


   // printf("iterations: %3d %3d \n", batch_iteration, online_iteration);
      
   // write to output array
    copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_final);    
    
    free(dist);
    free(cluster_assignment_cur);
    free(cluster_assignment_prev);
    free(point_move_score);
}           
           

#define CLUSTERS (3)

void main(void)
{
    int dim=1;
    int row, col, i;
    double floatGrayVal[IMG_HEIGHT*IMG_WIDTH];
    double cluster_centroid[CLUSTERS*2];
    int cluster_assignment[IMG_HEIGHT*IMG_WIDTH];

    printf("K-means test\n");

    for(row=0; row<IMG_HEIGHT; row++)
    {
        for(col=0; col<IMG_WIDTH; col++)
        {
           floatGrayVal[(row*IMG_WIDTH)+col]=(double)grayVal[(row*IMG_WIDTH)+col];
        }
    }

    // guess at sun location
    cluster_centroid[0]=10.0;
    cluster_centroid[1]=20.0;
    // guess at sky location
    cluster_centroid[2]=5.0;
    cluster_centroid[3]=20.0;
    // guess at water location
    cluster_centroid[4]=25.0;
    cluster_centroid[5]=20.0;

    kmeans(dim,		                // dimension of data 
           floatGrayVal,                // pointer to data
           (IMG_HEIGHT*IMG_WIDTH),      // number of elements
           CLUSTERS,                    // land, water, sky, sun
           cluster_centroid,            // initial cluster centroids
           cluster_assignment           // output
           );

    printf("K-means test done\n");
}
