import os
import time
import argparse
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

def visualize_point_clouds(input_dir: str, output_video: str = None, fps: int = 30):
    """
    Visualize point clouds from a directory sequentially and optionally save to video
    Args:
        input_dir: Directory containing point cloud files
        output_video: Path to save output video (optional)
        fps: Frames per second for visualization
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Initialize empty point cloud
    pointcloud = o3d.geometry.PointCloud()
    vis.add_geometry(pointcloud)
    
    # Get sorted list of files
    input_path = Path(input_dir)
    files = sorted([f for f in input_path.glob("*.ply")])
    
    if len(files) == 0:
        print(f"No .ply files found in {input_dir}")
        return
        
    print(f"Found {len(files)} point cloud files")
    
    # Frame timing
    frame_time = 1.0 / fps
    to_reset = True
    
    # Setup video writer if output path is provided
    video_writer = None
    if output_video:
        window_width = 1920
        window_height = 1080
        vis.create_window(width=window_width, height=window_height)
        video_writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (window_width, window_height)
        )
    
    try:
        for f in files:
            start_time = time.time()
            
            # Load point cloud
            try:
                pcd = o3d.io.read_point_cloud(str(f))
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
                
            # Update points and colors
            pointcloud.points = pcd.points
            if pcd.has_colors():
                pointcloud.colors = pcd.colors
                
            # Update visualization
            vis.update_geometry(pointcloud)
            if to_reset:
                vis.reset_view_point(True)
                to_reset = False
                
            vis.poll_events()
            vis.update_renderer()
            
            # Capture and save frame if recording
            if video_writer:
                # Capture frame from visualizer
                image = vis.capture_screen_float_buffer()
                # Convert to uint8 and BGR format
                image = np.asarray(image) * 255
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(image)
            
            # Control frame rate
            process_time = time.time() - start_time
            if process_time < frame_time:
                time.sleep(frame_time - process_time)
                
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        if video_writer:
            video_writer.release()
        vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sequence of point clouds")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing .ply files")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Path to save output video (optional)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for visualization")
    
    args = parser.parse_args()
    visualize_point_clouds(args.input_dir, args.output_video, args.fps)
