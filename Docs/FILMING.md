# Filming for FreeClipMocap

FreeClipMocap reads your motion from one ordinary video. What the camera sees decides how good the animation is.

## The camera

- **Keep the camera still.** Use a tripod or lean the phone against something. A moving camera looks like a moving
  actor: planted feet are found by not moving in the picture.
- **Film level, at about hip to chest height.** Slight tilts are corrected; strong angles from above or below are not.
- **Film at 1080p or 4K.** At 720p a full-body shot leaves too few pixels for the fingers.
- **60 fps for fast moves** (swings, kicks, spins); 30 fps is fine for slow ones. Fast motion blurs at low frame rates.

## Framing

- **The whole body in the picture, feet included, for the whole take.** Body parts that are rarely visible are left
  unanimated, and feet that leave the picture cannot be planted.
- **When the fingers matter, make the hands big.** Each hand should be at least about 120 px from wrist to fingertip in
  the video: at 4K a full-body shot gives that; at 1080p, film closer (knees up) or film the hands separately. At 720p
  with the full body in the picture, hands are about 60 px and the fingers come out jittery.
- **Face the camera, or at most a three-quarter turn.** Turning away hides arms and hands, and twists toward or away
  from the camera are hard to read from one view.

## The actor and the room

- **Even light, no strong backlight.** A window behind the actor turns the body into a silhouette.
- **Clothes that show the body's shape** and stand out from the background. Loose sleeves hide the elbows and wrists.
- **Leave room around the actor.** Steps toward or away from the camera flatten (depth is the weakest direction).
- **Start and end with a second of stillness.** It gives clean first and last frames; trim them later with the
  Start/End frames in the panel.

## Two phones (coming: not in the panel yet)

- **Print the board** (saved by the extension) on A4 or Letter, tape it flat on cardboard cut flush with the paper's bottom edge, and measure one square with a ruler.
  **Or show it on a tablet** (full-screen, brightness up, no window reflecting in it): measure one square on the screen, pick Paper: Custom, and enter Below Squares, the distance from the lowest row of squares down to the tablet's bottom edge.
- **Phones on stands, 60–90° apart,** both seeing the whole body. They must not move between calibration and the takes they are used for.
- **Turn video stabilisation off** on both phones; it shifts the picture from frame to frame.
- **Clap once after both phones are recording.** The clap lines the videos up.
- **Calibrate:** hold the board close to each phone in turn (about 1 m) and tilt it in several directions, then move it slowly where both phones see it, turned toward the space between them, at 1080p within about 2.5 m. 4K reaches about twice as far.
- **Know your phone's focal length?** Pick Lens: Typed and enter it (the 35 mm equivalent, as spec sheets give it; for example 23 mm). With Keep Fixed off, the board still corrects it, which is safest: spec values are rounded and meant for photos. With Keep Fixed on it is used exactly and no close-ups are needed. After a calibration each camera row shows the measured focal length to compare.
- **Save each phone's lens once:** after a good calibration, press the save button next to each camera's Lens menu and give it a name (for example "Pixel 8 main 1080p"). Later calibrations with that lens picked skip the close-ups: only the part where both phones see the board is needed. A saved lens holds for the same phone, the same lens (main or ultra-wide), resolution, orientation (portrait or landscape) and zoom, with stabilisation off; change any of these and measure again.
- **No board? Calibrate from the body** (Method: Body). Measure the distance between the two phones' lenses with a tape and enter it as Phones Apart, and give each camera its focal length (Lens: Typed) or a saved lens. Then, with both phones recording and after the clap, walk around the whole capture area for 20–30 seconds: stop for a moment at several spots (the floor comes from the feet standing still), turn, and move your arms. Less accurate than the board: the cameras come out a few cm off at best, and a focal length that is a few % off adds several cm more, so a lens saved from one board calibration helps most.
- **Floor:** stand the board on its bottom edge on the floor, facing between the phones, still for about 2 seconds (lean it against a box). Without this the floor is taken from the feet: Capture the take before pressing Calibrate (or press Calibrate again after capturing), and stand still with both feet down for a moment somewhere in the take.
