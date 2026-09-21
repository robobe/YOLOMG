# How YOLOMG sees motion

Imagine you are watching a drone video. A normal object detector sees one picture at a time. YOLOMG gets two pictures:

1. The normal color video frame.
2. A special black-and-white-ish **motion picture** that says, “which pixels changed in an interesting way?”

The second picture helps because a drone can be tiny and hard to see in the normal image, but it often moves differently from the background.

## Why plain frame difference is not enough

A first idea is simple:

```text
motion picture = current frame - previous frame
```

This works if the camera is standing still. But a drone camera also moves. If the camera turns left, **every building, road, and tree moves on the screen**. A plain difference image would light up almost the whole picture. That is noisy and not helpful.

## The key idea: cancel the camera movement first

YOLOMG tries to answer this question:

> “How would the old frame look if the camera had already moved into the current camera position?”

It does this using background details such as building corners and road markings.

```text
old frame  -- adjust for camera movement --> adjusted old frame
current frame -----------------------------> current frame

motion image = difference between these two
```

After the camera movement is cancelled, a stationary building should line up with itself and become dark in the difference image. A moving drone often does not line up, so it stays bright.

## Step by step

The original `mask32` generator uses three frames around the frame it wants to detect.

```text
frame t-2          frame t          frame t+2
  earlier          target frame       later
     \                 |                /
      \                |               /
       camera-motion compensation + differences
                        |
                        v
                  final motion mask
```

### Step 1: choose the target frame

Suppose we want to detect a drone in frame `t`. Frame `t` is the normal RGB image sent to the detector and the frame with the annotation box.

### Step 2: make the images easier to compare

The code blurs every frame a little and converts it to grayscale. Blur removes tiny camera noise. Grayscale removes color because motion is mostly about position, not color.

### Step 3: find stable background points

The program places a grid of points over the image and uses KLT optical flow to track where those points move between frames.

Think of it as putting many tiny stickers on buildings and roads, then asking where each sticker went in the next image.

### Step 4: estimate the camera movement

Some tracked points can be wrong, especially points on the flying drone. RANSAC ignores many bad matches and uses the most consistent background matches to estimate one camera transform called a **homography**.

A homography can represent perspective changes, rotation, and translation. It is like stretching and sliding the earlier image until its background matches the target image.

## How camera movement is ignored in detail

“Ignoring camera movement” does not mean the program knows the real 3D camera position. It means the program finds a **2D picture transform** that makes the background in two pictures line up as closely as possible.

### A simple camera-pan example

Imagine the camera moves right. On the screen, a stationary building appears to move left by 20 pixels:

```text
earlier frame:       building corner at x = 500
target frame:        same corner at x = 480
```

The program sees that many background points changed in a similar way. It learns a rule like:

```text
“move the earlier picture left by 20 pixels to match the target picture”
```

After applying that rule, the building corner is at `x = 480` in both images. Its difference becomes close to zero, so it is dark in the motion mask.

A drone can have its own movement on top of the camera movement:

```text
background after camera correction:  matches
drone after camera correction:       still does not match
```

That leftover mismatch is the bright evidence that helps the detector.

### 1. KLT optical flow finds where the scene moved

The code creates a regular grid of points over the grayscale image. For each point, KLT optical flow looks at the small pixel pattern around it and searches for the most similar pattern in the next frame.

```text
frame t-2                         frame t
  point A at (x, y)  -------->      point A at (x-20, y+2)
```

Buildings, roads, and large background objects give reliable points because their small patterns are easy to follow. A blank sky often gives unreliable points because it has almost no texture.

### 2. RANSAC chooses the shared background movement

Not every tracked point should control the camera transform:

- A point on the drone moves differently.
- A point on a moving car moves differently.
- A blurry or repeated texture can be tracked incorrectly.

RANSAC repeatedly tries a transform from a small set of matches, then counts how many other matches agree with it. The transform supported by the largest group wins.

```text
background matches:  mostly agree with one camera movement  -> keep
drone / car matches: disagree with that movement            -> ignore
bad matches:         disagree                              -> ignore
```

This is why the method can use many point tracks without accidentally deciding that the drone's movement is the camera movement.

### 3. A homography is the chosen camera-picture transform

The winning transform is a 3×3 matrix called `H`:

```text
[x_target]       [ h11 h12 h13 ] [x_source]
[y_target]   ~   [ h21 h22 h23 ] [y_source]
[   1    ]       [ h31 h32 h33 ] [   1    ]
```

You do not need to calculate this matrix by hand. The important idea is that it can move, rotate, scale, skew, and slightly change perspective across the whole image. It is stronger than using only one average left/right shift.

### 4. Warp the source image into the target camera view

OpenCV uses `warpPerspective` with `H` to create a new image:

```text
source frame                         compensated source frame
before camera correction      ->     looks like target camera view
```

Now, most stationary background pixels should have the same coordinates as in the target frame. Subtracting them mostly produces black pixels.

### 5. Use only the leftover error as motion evidence

The important operation is:

```text
residual = absolute_value(target_frame - compensated_source_frame)
```

This residual contains three kinds of brightness:

1. **Useful:** independently moving objects, including a drone.
2. **Normal error:** small alignment mistakes, blur, shadows, or exposure changes.
3. **Bad error:** places where the camera movement model is not good enough.

The neural network learns to use the useful patterns together with RGB appearance. It does not treat every bright pixel as a drone.

## Limits of camera-motion compensation

The method reduces camera motion; it cannot remove it perfectly.

| Situation | Why it can leave bright noise |
| --- | --- |
| Camera moves sideways near close buildings | Close and far objects move by different amounts (parallax), but one homography describes only one global transform. |
| Motion blur or fast rotation | KLT points become hard to track correctly. |
| Mostly blank sky or water | There are too few textured background points. |
| Large moving foreground object | It can hide background points and reduce RANSAC quality. |
| Image borders after warping | Some pixels have no matching source pixel after the frame is shifted or rotated. |

The current code falls back to an almost-identity transform when it has fewer than 15 tracked points. In that case, strong camera motion may remain in the motion mask.

The compensation function calculates an invalid-border mask, but the original `FD5_mask.py` does not currently subtract that mask from its final residual. For a faithful baseline, keep this behavior. For a later improved version, masking those invalid borders is a sensible experiment because it can remove bright warp-edge artifacts.

### Step 5: warp the outer frames into the target view

The earlier frame `t-2` is warped to look like it was filmed from the camera position at `t`. The later frame `t+2` is warped in the same way.

```text
warp(t-2) -> view of t
warp(t+2) -> view of t
```

### Step 6: calculate two difference images

For each warped frame, the code calculates the absolute pixel difference from the target frame:

```text
previous_difference = abs(frame_t - warp(frame_t-2))
future_difference   = abs(frame_t - warp(frame_t+2))
```

Bright pixels mean “this area still changed even after we tried to cancel camera movement.” A moving drone should create a bright small region.

### Step 7: average the two differences

The final mask is the average of the differences from the earlier and later frames:

```text
motion_mask = (previous_difference + future_difference) / 2
```

Using both sides of the target frame is more stable than relying on only one neighbor. For example, if blur or a bad match harms one comparison, the other comparison can still help.

## What the detector learns

The detector receives:

```text
RGB frame:     “What does this look like?”
Motion mask:   “What changed here?”
```

The attention fusion layer combines both clues. It can learn that a tiny dark dot in the sky is more likely to be a drone when the same place is also bright in the motion mask.

## A practical tradeoff

The original method needs `t+2`, a future frame. That means a live video program must wait for two extra frames before it can show the result for frame `t`.

At 30 FPS, two frames are about 67 milliseconds. This is usually acceptable for offline video processing and many drone applications, but it is not zero-latency.

## Important rule for YOLO26 migration

If we train a new YOLO26 model with this motion-mask method, the video application must use the **same** method. Training with compensated three-frame masks but testing with a simple previous-frame difference would confuse the model because its second input would look different from what it learned.

## Is motion used only for training?

No. Motion is used in **both training and inference**.

During training, the model sees RGB frames, motion masks, and the correct drone boxes. It learns how to use both images together.

During inference, there are no correct boxes because the program is trying to find them. The video program still creates the motion mask and sends it with the RGB frame into the model. The model then predicts the boxes by itself.

```text
TRAINING
RGB frame + motion mask + teacher's correct drone box
                         |
                         v
                model learns from mistakes

INFERENCE
RGB frame + motion mask
                         |
                         v
                 model predicts drone box
```

If we gave the model only RGB during inference, it would be missing half of the information it learned to expect. It might still run only after redesigning the model, but it would not be the same trained YOLOMG model.

## What do the model inputs look like?

Computers store an image as a stack of color channels. One normal color image has three channels:

```text
red, green, blue
```

YOLOMG has two separate image inputs with the same size:

```text
rgb input:          [batch, 3, height, width]
motion-mask input:  [batch, 3, height, width]
```

For one 1280 by 1280 image, the shapes look like this:

```text
RGB:    [1, 3, 1280, 1280]
motion: [1, 3, 1280, 1280]
```

`1` means one example at a time. The motion mask starts as a grayscale image, but it is read as three matching channels so it has the same input shape as a normal image.

The current model does **not** simply glue the six channels together at the first pixel. Instead it has two small starting paths:

```text
RGB frame       -> RGB stem       -> RGB features
motion mask     -> motion stem    -> motion features
                                      |
                                      v
                              attention fusion (Concat3)
                                      |
                                      v
                              detector predicts boxes
```

This is useful because RGB and motion mean different things. RGB gives appearance clues such as shape and color. Motion gives change clues. The attention fusion decides which locations and feature channels deserve more attention.

## What happens during video inference?

Here is the full process for target frame `t`:

1. The program reads video frames until it has `t-2`, `t`, and `t+2`.
2. It creates the camera-motion-compensated mask for frame `t`.
3. It resizes and normalizes the RGB image and the motion mask in the same way.
4. It sends both tensors into the model: `model(rgb, motion_mask)`.
5. The two stems and attention fusion make combined features.
6. The detector predicts many possible boxes, scores, and class names.
7. Non-maximum suppression removes duplicate boxes that describe the same drone.
8. The remaining boxes are resized back to the original video resolution, drawn on frame `t`, and written to the output MP4.

```text
video frames -> compensated motion mask -> model(RGB, mask)
                                                  |
                                                  v
                                      boxes + confidence scores
                                                  |
                                                  v
                              draw boxes and FPS on output video
```

Because the method waits for frame `t+2`, the output is shown two frames after frame `t` was captured. It is still processing every frame; it is simply using a tiny look-ahead buffer to make better motion information.

## Summary

- The motion mask is a second image, not a training-only trick.
- Training teaches the model how RGB and motion work together.
- Inference must create the same kind of motion mask and pass both inputs to the model.
- The two inputs are processed separately first, then fused with attention.
- Camera-motion compensation stops the moving camera from making the whole scene look like a moving object.
- The original three-frame method gives cleaner motion evidence but adds a two-frame delay.
