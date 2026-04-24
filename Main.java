import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.Random;
import lib.ai.network.*;

public class Main extends JPanel implements ActionListener, KeyListener {
    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;

    private Rectangle bird;
    private ArrayList<Rectangle> pipes;
    private Timer gameLoop;
    private Random random;

    private int birdVelocity = 0;
    private final int gravity = 1;
    private int score = 0;
    private boolean gameOver = false;
    private boolean gameStarted = false;

    private Model ai = new Model(new int[]{4, 8, 2}, 0.01); 
    private int lastAction = 0; 

    public Main() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(new Color(135, 206, 235));
        setFocusable(true);
        addKeyListener(this);

        random = new Random();
        bird = new Rectangle(WIDTH / 2 - 10, HEIGHT / 2 - 10, 20, 20);
        pipes = new ArrayList<>();

        spawnPipe();
        spawnPipe();

        gameLoop = new Timer(33, this);
        gameLoop.start();
    }

    private void spawnPipe() {
        int width = 80;
        int gap = 200;
        int minHeight = 50;
        int maxHeight = HEIGHT - gap - minHeight - 100;
        int h = minHeight + random.nextInt(maxHeight);
        int x = pipes.isEmpty() ? WIDTH : pipes.get(pipes.size() - 1).x + 350;

        pipes.add(new Rectangle(x, 0, width, h)); 
        pipes.add(new Rectangle(x, h + gap, width, HEIGHT - h - gap)); 
    }

    private double[] getCurrentState() {
        Rectangle nextPipe = pipes.get(0);

        if (nextPipe.x + nextPipe.width < bird.x) {
            nextPipe = pipes.size() > 2 ? pipes.get(2) : pipes.get(0);
        }

        return new double[]{
            bird.y / (double)HEIGHT,               
            birdVelocity / 20.0,                 
            (nextPipe.x - bird.x) / (double)WIDTH, 
            nextPipe.height / (double)HEIGHT     
        };
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (gameStarted && !gameOver) {

            double[] currentState = getCurrentState();
            double reward = 0.1; 

            double[] output = ai.predict(currentState);
            lastAction = (output[0] > 0.5) ? 1 : 0;

            if (lastAction == 1) jump();

            birdVelocity += gravity;
            bird.y += birdVelocity;

            for (int i = pipes.size() - 1; i >= 0; i--) {
                Rectangle pipe = pipes.get(i);
   
                pipe.x -= 4; 

                if (pipe.y == 0 && bird.x > pipe.x + pipe.width && bird.x <= pipe.x + pipe.width + 4) {
                    score++;
                    reward = 1.0;
                }

                if (pipe.x + pipe.width < 0) {
                    pipes.remove(i);
                    if (pipe.y == 0) spawnPipe();
                }

                if (pipe.intersects(bird)) {
                    gameOver = true;
                    reward = -10.0; 
                }
            }

            if (bird.y > HEIGHT || bird.y < 0) {
                gameOver = true;
                reward = -10.0; 
            }

   
            double[] target = output.clone();
            double[] nextState = getCurrentState();
            double[] nextOutput = ai.predict(nextState);

            double gamma = 0.9;
            target[0] = reward + (gamma * Math.max(nextOutput[0], 0));

            for(int i = 0; i < 10; i++){
                ai.train(currentState, target);
            }
        }

        if (gameOver) {
            restart();
        }

        repaint();
    }

    private void jump() {
        if (!gameOver) {
            if (!gameStarted) gameStarted = true;
            birdVelocity = -5;
        }
    }

    private void restart() {
        bird = new Rectangle(WIDTH / 2 - 10, HEIGHT / 2 - 10, 20, 20);
        birdVelocity = 0;
        score = 0;
        gameOver = false;
        gameStarted = true; 
        pipes.clear();
        spawnPipe();
        spawnPipe();
    }

    @Override protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setColor(new Color(34, 139, 34));
        for (Rectangle pipe : pipes) g2.fillRect(pipe.x, pipe.y, pipe.width, pipe.height);
        g2.setColor(Color.YELLOW);
        g2.fillRect(bird.x, bird.y, bird.width, bird.height);
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("Arial", Font.BOLD, 50));
        g2.drawString(String.valueOf(score), WIDTH / 2 - 20, 80);
    }

    @Override public void keyPressed(KeyEvent e) { if (e.getKeyCode() == KeyEvent.VK_SPACE) jump(); }
    @Override public void keyReleased(KeyEvent e) {}
    @Override public void keyTyped(KeyEvent e) {}

    public static void main(String[] args) {
        JFrame frame = new JFrame("AI Flappy Bird");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new Main());
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}